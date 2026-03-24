# 统一使用高版本 TensorFlow 内置 Keras 接口
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Bidirectional, Layer, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import KerasTensor


# 手动实现维特比解码算法（统一数据类型，解决类型冲突+索引匹配）
def viterbi_decode(emissions, transition_params, static_seq_len):
    """
    维特比解码：统一数据类型为int64，适配 tf.argmax 返回类型，索引统一为int32
    :param emissions: 单个样本的发射概率，形状 (static_seq_len, num_cates)
    :param transition_params: 状态转移矩阵，形状 (num_cates, num_cates)
    :param static_seq_len: 静态固定序列长度（如84）
    :return: (最优标签序列, 序列对应的对数概率)
    """
    num_cates = tf.shape(emissions)[1]
    seq_len = static_seq_len

    # 修正1：TensorArray 存储数据为int64（匹配tf.argmax），索引必须为int32
    backpointers = tf.TensorArray(dtype=tf.int64, size=seq_len, dynamic_size=False)
    prev_score = emissions[0]  # 初始前向得分

    # 符号化循环计算前向得分和回溯指针
    def loop_body(t, prev_score, backpointers):
        candidate_scores = tf.expand_dims(prev_score, axis=1) + transition_params
        current_score = tf.reduce_max(candidate_scores, axis=0) + emissions[t]
        current_backpointer = tf.argmax(candidate_scores, axis=0)  # 返回int64
        # 修正2：索引t转为int32，匹配TensorArray.write要求
        t_int32 = tf.cast(t, dtype=tf.int32)
        backpointers = backpointers.write(t_int32, current_backpointer)
        return t + 1, current_score, backpointers

    # 修正3：循环变量t改为int32类型，避免索引类型冲突
    _, final_prev_score, final_backpointers = tf.while_loop(
        cond=lambda t, *args: t < seq_len,
        body=loop_body,
        loop_vars=(tf.constant(1, dtype=tf.int32), prev_score, backpointers),
        parallel_iterations=1,
        back_prop=True
    )

    # 回溯获取最优标签序列
    final_tag = tf.cast(tf.argmax(final_prev_score, axis=0), dtype=tf.int64)
    final_tag = tf.squeeze(final_tag)

    # 修正4：viterbi_path 存储数据为int64，索引为int32
    viterbi_path = tf.TensorArray(dtype=tf.int64, size=seq_len, dynamic_size=False)
    # 修正5：写入最后一个索引时转为int32
    viterbi_path = viterbi_path.write(tf.cast(seq_len - 1, dtype=tf.int32), final_tag)
    current_tag = final_tag

    # 符号化回溯循环（修正索引读取逻辑）
    def backtrack_loop_body(t, current_tag, viterbi_path):
        # 修正6：读取回溯指针时，索引转为int32，避免越界
        t_int32 = tf.cast(t, dtype=tf.int32)
        current_backpointer = final_backpointers.read(t_int32)
        current_tag = tf.gather(current_backpointer, current_tag)
        # 修正7：写入时索引转为int32
        viterbi_path = viterbi_path.write(t_int32, current_tag)
        return t - 1, current_tag, viterbi_path

    # 修正8：回溯循环变量t改为int32类型
    _, _, final_viterbi_path = tf.while_loop(
        cond=lambda t, *args: t >= 0,
        body=backtrack_loop_body,
        loop_vars=(tf.cast(seq_len - 2, dtype=tf.int32), current_tag, viterbi_path),
        parallel_iterations=1,
        back_prop=True
    )

    # 强制塑形并转换为与标签一致的int32类型
    viterbi_path = final_viterbi_path.stack()
    viterbi_path = tf.reshape(viterbi_path, (seq_len,))
    viterbi_path = tf.cast(viterbi_path, dtype=tf.int32)
    final_score = tf.reduce_max(final_prev_score)

    return viterbi_path, final_score


# 自定义 CRF 层（统一数据类型+符号化操作，兼容图模式）
class CustomCRF(Layer):
    def __init__(self, num_cates, static_seq_len, sparse_target=True, **kwargs):
        """
        初始化自定义CRF层
        :param num_cates: 实体类别数量（即标签数量）
        :param static_seq_len: 静态固定序列长度（如84）
        :param sparse_target: 是否为稀疏标签（固定为True）
        :param kwargs: 父类Layer的额外参数
        """
        self.num_cates = num_cates
        self.static_seq_len = static_seq_len
        self.sparse_target = sparse_target
        self.transition_params = None
        kwargs.setdefault('dtype', tf.float32)
        # 明确输出规格，避免Keras无法识别张量
        super(CustomCRF, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建可训练参数：状态转移矩阵（强制触发，确保参数初始化）
        """
        # 增加输入形状校验，确保build方法被正确触发
        if len(input_shape) != 3:
            raise ValueError(f"CRF层输入必须为三维张量 (batch, seq_len, num_cates)，当前为 {input_shape}")
        assert input_shape[-1] == self.num_cates, f"CRF输入最后一维必须为类别数{self.num_cates}，当前为{input_shape[-1]}"

        self.transition_params = self.add_weight(
            name='transition_params',
            shape=(self.num_cates, self.num_cates),
            initializer='glorot_uniform',
            trainable=True
        )
        super(CustomCRF, self).build(input_shape)
        # 标记层已构建，避免重复初始化
        self.built = True

    def call(self, inputs, training=None):
        """
        前向传播：明确训练/推理阶段返回值，确保张量有效（全符号化操作）
        :param inputs: 三维发射概率张量 (batch, static_seq_len, num_cates)
        :param training: 是否为训练模式
        """
        # 强制初始化build（若未自动触发）
        if not self.built:
            self.build(inputs.shape)

        # 统一training参数默认值（符号化赋值）
        training = tf.constant(True) if training is None else training

        # 训练阶段：返回三维发射概率张量（供crf_loss计算损失，保持原形状）
        if training:
            # 确保返回值为float32，与层dtype一致
            emissions = tf.cast(inputs, dtype=self.dtype)
            return emissions

        # 推理阶段：返回二维标签序列张量（维特比解码结果）
        else:
            batch_size = tf.shape(inputs)[0]

            def batch_viterbi(i):
                # 修正：索引i转为int32（匹配TensorArray索引要求+图模式兼容）
                i = tf.cast(i, dtype=tf.int32)
                single_input = tf.gather(inputs, i, axis=0)
                single_input = tf.reshape(single_input, (self.static_seq_len, self.num_cates))
                viterbi_seq, _ = viterbi_decode(
                    single_input,
                    self.transition_params,
                    self.static_seq_len
                )
                return tf.reshape(viterbi_seq, (self.static_seq_len,))

            # 修正：tf.range返回int32类型，匹配batch_viterbi中的索引类型
            viterbi_sequences = tf.map_fn(
                fn=batch_viterbi,
                elems=tf.range(batch_size, dtype=tf.int32),
                fn_output_signature=tf.TensorSpec(shape=(self.static_seq_len,), dtype=tf.int32)
            )
            return viterbi_sequences

    def compute_output_shape(self, input_shape):
        """
        固定输出形状
        """
        return (input_shape[0], self.static_seq_len)

    def compute_output_spec(self, input_spec):
        """
        固定输出规格
        """
        input_shape = input_spec.shape
        output_shape = self.compute_output_shape(input_shape)
        return KerasTensor(
            shape=output_shape,
            dtype=tf.int32,
            name=f"{self.name}_output"
        )

    @tf.function(reduce_retracing=True)
    def compute_loss(self, y_true, y_pred):
        """
        计算 CRF 对数似然损失（全符号化操作，兼容图模式）
        """
        # 步骤1：强制统一张量形状和数据类型（符号化操作，无Python判断）
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # 符号化挤压多余维度，无需if判断
        y_true = tf.squeeze(y_true, axis=-1)
        batch_size = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, (batch_size, self.static_seq_len))

        # 步骤2：初始化序列长度（符号化张量，无Python判断）
        sequence_lengths = tf.ones(shape=(batch_size,), dtype=tf.int32) * self.static_seq_len

        # 步骤3：单个样本对数似然计算（全符号化）
        def single_sample_log_likelihood(args):
            y_true_single, y_pred_single, seq_len_single = args

            # 符号化重塑形状，无需if判断
            y_true_single = tf.reshape(y_true_single, (self.static_seq_len,))
            y_pred_single = tf.reshape(y_pred_single, (self.static_seq_len, self.num_cates))
            seq_len_single = tf.cast(tf.squeeze(seq_len_single), dtype=tf.int32)

            # 符号化提取有效序列
            valid_indices = tf.range(seq_len_single, dtype=tf.int32)
            y_true_valid = tf.gather(y_true_single, valid_indices)
            y_pred_valid = tf.gather(y_pred_single, valid_indices, axis=0)

            # 子步骤1：计算真实路径得分
            start_tag = tf.gather(y_true_valid, 0)
            start_tag = tf.squeeze(start_tag)
            start_emission = tf.gather(y_pred_valid, 0, axis=0)
            initial_score = tf.gather(start_emission, start_tag)

            def accumulate_loop(t, prev_score, prev_tag):
                # 统一循环变量和索引类型为int32（符号化）
                t = tf.cast(t, dtype=tf.int32)
                curr_tag = tf.squeeze(tf.gather(y_true_valid, t))
                prev_tag = tf.squeeze(tf.gather(y_true_valid, t - 1))

                curr_emission = tf.gather(y_pred_valid, t, axis=0)
                curr_emission_score = tf.gather(curr_emission, curr_tag)
                transition_score = tf.gather(tf.gather(self.transition_params, prev_tag), curr_tag)

                new_score = prev_score + curr_emission_score + transition_score
                return t + 1, new_score, curr_tag

            # 执行符号化循环
            _, final_path_score, _ = tf.while_loop(
                cond=lambda t, *args: t < seq_len_single,
                body=accumulate_loop,
                loop_vars=(tf.constant(1, dtype=tf.int32), initial_score, start_tag),
                parallel_iterations=1,
                back_prop=True
            )

            # 子步骤2：计算所有路径总得分（符号化条件分支用tf.cond）
            initial_forward_score = tf.gather(y_pred_valid, 0, axis=0)

            def forward_step(prev_score, curr_emission):
                candidate_scores = tf.expand_dims(prev_score, axis=1) + self.transition_params
                return tf.reduce_logsumexp(candidate_scores, axis=0) + curr_emission

            # 符号化提取后续时刻发射概率
            subsequent_indices = tf.range(1, self.static_seq_len, dtype=tf.int32)
            subsequent_emissions = tf.gather(y_pred_valid, subsequent_indices, axis=0)
            forward_scores = tf.scan(
                fn=forward_step,
                elems=subsequent_emissions,
                initializer=initial_forward_score
            )

            # 符号化条件分支，替代Python if/else
            final_forward_score = tf.cond(
                pred=tf.greater(seq_len_single, 1),
                true_fn=lambda: tf.gather(forward_scores, seq_len_single - 2),
                false_fn=lambda: initial_forward_score
            )
            total_score = tf.reduce_logsumexp(final_forward_score, axis=0)

            # 对数似然计算
            return final_path_score - total_score

        # 步骤4：批量处理所有样本（符号化操作）
        log_likelihoods = tf.map_fn(
            fn=single_sample_log_likelihood,
            elems=(y_true, y_pred, sequence_lengths),
            fn_output_signature=tf.float32
        )

        # 步骤5：返回批量平均损失（标量张量）
        return -tf.reduce_mean(log_likelihoods)

    @tf.function(reduce_retracing=True)
    def compute_accuracy(self, y_true, y_pred):
        """
        计算准确率（全符号化操作，兼容图模式，无Python原生判断）
        """
        batch_size = tf.shape(y_pred)[0]

        # 批量预测（统一索引类型为int32，兼容图模式）
        def batch_predict(i):
            # 修正：索引i转为int32，匹配TensorArray索引要求
            i = tf.cast(i, dtype=tf.int32)
            single_pred = tf.gather(y_pred, i, axis=0)
            single_pred = tf.reshape(single_pred, (self.static_seq_len, self.num_cates))
            viterbi_seq, _ = viterbi_decode(
                single_pred,
                self.transition_params,
                self.static_seq_len
            )
            return tf.reshape(viterbi_seq, (self.static_seq_len,))

        # 修正：tf.range返回int32类型，匹配batch_predict中的索引类型
        y_pred_labels = tf.map_fn(
            fn=batch_predict,
            elems=tf.range(batch_size, dtype=tf.int32),
            fn_output_signature=tf.TensorSpec(shape=(self.static_seq_len,), dtype=tf.int32)
        )

        # 符号化统一形状和数据类型，无需Python判断
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.reshape(y_true, (batch_size, self.static_seq_len))

        correct_predictions = tf.equal(y_true, y_pred_labels)
        return tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))

    def get_config(self):
        """
        保存层配置（兼容模型保存与加载）
        """
        config = super(CustomCRF, self).get_config()
        config.update({
            'num_cates': self.num_cates,
            'static_seq_len': self.static_seq_len,
            'sparse_target': self.sparse_target
        })
        return config


# 自定义 CRF 损失包装器（全符号化操作，移除Python原生判断）
def crf_loss(crf_layer):
    def loss(y_true, y_pred):
        """
        包装CRF损失函数，确保返回有效标量损失（兼容图模式）
        :param y_true: 二维标签张量 (batch, static_seq_len)
        :param y_pred: 三维发射概率张量 (batch, static_seq_len, num_cates)
        """
        # 步骤1：符号化挤压多余维度（无需if判断，tf.squeeze兼容无多余维度场景）
        y_true = tf.squeeze(y_true, axis=-1)

        # 步骤2：符号化确保张量形状（校验+适配，无Python判断）
        y_true = tf.ensure_shape(y_true, (None, crf_layer.static_seq_len))
        y_pred = tf.ensure_shape(y_pred, (None, crf_layer.static_seq_len, crf_layer.num_cates))

        # 步骤3：符号化断言张量秩（调试用，兼容图模式）
        tf.debugging.assert_rank(y_true, 2, message="y_true 必须是二维张量 (batch, static_seq_len)")
        tf.debugging.assert_rank(y_pred, 3, message="y_pred 必须是三维张量 (batch, static_seq_len, num_cates)")

        # 步骤4：调用CRF层损失计算，确保返回标量
        crf_loss_value = crf_layer.compute_loss(y_true, y_pred)
        crf_loss_value = tf.ensure_shape(crf_loss_value, ())  # 确保为标量张量

        return crf_loss_value

    return loss


# 自定义 CRF 准确率包装器（全符号化操作，兼容图模式）
def crf_accuracy(crf_layer):
    def accuracy(y_true, y_pred):
        """
        包装CRF准确率函数，确保返回有效标量准确率（兼容图模式）
        """
        # 符号化挤压+形状确保，无需Python判断
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.ensure_shape(y_true, (None, crf_layer.static_seq_len))
        y_pred = tf.ensure_shape(y_pred, (None, crf_layer.static_seq_len, crf_layer.num_cates))

        # 调用CRF层准确率计算，确保返回标量
        crf_acc_value = crf_layer.compute_accuracy(y_true, y_pred)
        crf_acc_value = tf.ensure_shape(crf_acc_value, ())

        return crf_acc_value

    return accuracy


# 重构 LSTM-CRF 模型构建函数（保持维度映射，统一数据类型，验证参数初始化）
def build_lstm_crf_model(num_cates, static_seq_len, vocab_size, model_opts=dict()):
    """
    构建 LSTM-CRF 模型（解决所有类型冲突+图模式兼容）
    :param num_cates: 类别数量
    :param static_seq_len: 静态固定序列长度（如84）
    :param vocab_size: 词汇表大小
    :param model_opts: 模型配置参数
    :return: 编译好的模型
    """
    # 模型默认配置
    opts = {
        'emb_size': 256,
        'emb_trainable': True,
        'emb_matrix': None,
        'lstm_units': 256,
        'optimizer': Adam(learning_rate=1e-4)
    }
    opts.update(model_opts)

    # 1. 输入层（指定int32类型，匹配输入数据）
    input_seq = Input(shape=(static_seq_len,), dtype=tf.int32)

    # 2. 嵌入层（维度映射，适配LSTM输入）
    if opts.get('emb_matrix') is not None:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=opts['emb_size'],
            weights=[opts['emb_matrix']],
            trainable=opts['emb_trainable']
        )
    else:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=opts['emb_size']
        )
    x = embedding(input_seq)

    # 3. 双向 LSTM 层（返回序列，适配序列标注任务）
    lstm = LSTM(opts['lstm_units'], return_sequences=True, recurrent_dropout=0.1)
    x = Bidirectional(lstm)(x)  # 输出形状：(None, static_seq_len, 2*lstm_units)

    # 4. 全连接层（维度映射到类别数，不可省略）
    x = Dense(num_cates, activation=None)(x)  # 输出形状：(None, static_seq_len, num_cates)

    # 5. 自定义 CRF 层（接收类别维度输入，形状匹配）
    crf = CustomCRF(num_cates, static_seq_len=static_seq_len, sparse_target=True)
    output = crf(x)

    # 6. 模型编译（使用修正后的损失和准确率函数）
    model = Model(input_seq, output)
    model.compile(
        optimizer=opts['optimizer'],
        loss=crf_loss(crf),
        metrics=[crf_accuracy(crf)]
    )

    # 验证CRF层参数是否已初始化（关键！避免转移矩阵为None）
    assert crf.transition_params is not None, "CRF层转移矩阵未初始化，构建模型失败"
    print("模型构建成功，CRF层转移矩阵已初始化")

    # 保存 CRF 层引用（方便后续预测和参数查看）
    model.crf_layer = crf
    return model