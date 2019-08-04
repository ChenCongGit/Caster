import string
import logging

import tensorflow as tf

from Caster.utils import shape_utils


class Label_map(object):
    def __init__(self,
                 character_set=None,
                 label_offset=0,
                 unk_label=None):
        if not isinstance(character_set, list):
            raise ValueError('character_set must be provided as a list')

        if len(frozenset(character_set)) != len(character_set):
            raise ValueError('Found duplicate characters in character_set')

        self._character_set = character_set
        self._label_offset = label_offset
        self._unk_label = unk_label or self._label_offset
        self._char_to_label_table, self._label_to_char_table = self._build_lookup_tables()

    @property
    def num_classes(self):
        return len(self._character_set)

    def _build_lookup_tables(self):
        """
        构建字符集映射表，真实字符与标签映射
        """
        chars = self._character_set
        labels = list(range(self._label_offset, self._label_offset + len(self._character_set))) # 空出0，1两个标签序号当作序列起始结束标志
        char_to_label_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(chars, labels, key_dtype=tf.string, value_dtype=tf.int64),
            default_value=self._unk_label
        )
        label_to_char_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(labels, chars, key_dtype=tf.int64, value_dtype=tf.string),
            default_value=""
        )
        return char_to_label_table, label_to_char_table

    def text_to_label(self, batch_text, return_dense=True, pad_value=-1, return_lengths=False):
        """
        给定字符型文本转化为整型标签序列，只适用于英文句子
        Args:
            text: ascii encoded string tensor with shape [batch_size]
            return_dense: whether to return dense labels
            pad_value: Value used to pad labels to the same length.
            return_lengths: if True, also return text lengths
        Returns:
            labels: sparse or dense tensor of labels
        """
        """
        # 英文句子按空格分词，例如source = ["hello world", "a b c"], delimiter='',
        # 返回tf.SparseTensor对象， st.indices = [0, 0; 0, 1; 1, 0; 1, 1; 1, 2]， 
        # st.shape = [2, 3] st.values = ['hello', 'world', 'a', 'b', 'c']
        # The first column of the indices corresponds to the row in source and the second column 
        # corresponds to the index of the split component in this row.
        """
        chars = tf.string_split(batch_text, delimiter='')
        labels_sp = tf.SparseTensor(
            chars.indices, self._char_to_label_table.lookup(chars.values), chars.dense_shape
        )

        if return_dense:
            labels = tf.sparse_tensor_to_dense(labels_sp, default_value=pad_value)
        else:
            labels = labels_sp

        if return_lengths:
            text_lengths = tf.sparse_reduce_sum(
                tf.SparseTensor(
                    chars.indices,
                    tf.fill([tf.shape(chars.indices)[0]], 1),
                    chars.dense_shape), axis=1
            )
            text_lengths.set_shape([None])
            return labels, text_lengths
        else:
            return labels

    def label_to_text(self, labels):
        """Convert labels to text strings.
        Args:
            labels: int32 tensor with shape [batch_size, max_label_length]
        Returns:
            text: string tensor with shape [batch_size]
        """
        if labels.dtype == tf.int32 or labels.dtype == tf.int64:
            labels = tf.cast(labels, tf.int64)
        else:
            raise ValueError('Wrong dtype of labels: {}'.format(labels.dtype))
        chars = self._label_to_char_table.lookup(labels) # [batch_size, max_label_length]

        """
        将张量中的字符拼接到一起
        # tensor `a` is [["a", "b"], ["c", "d"]]
        tf.reduce_join(a, 0) ==> ["ac", "bd"]
        tf.reduce_join(a, 1) ==> ["ab", "cd"]
        tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
        tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
        tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
        tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
        tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
        tf.reduce_join(a, [0, 1]) ==> "acbd"
        tf.reduce_join(a, [1, 0]) ==> "abcd"
        tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
        tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
        """
        batch_text = tf.reduce_join(chars, axis=1) # [batch_size]
        return batch_text


