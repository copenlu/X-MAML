
""" NLI processors and helpers """

import logging
import os
import csv
import sys
import copy
import json
from transformers.data.processors.utils import DataProcessor
from transformers.file_utils import is_tf_available
if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)





class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None,domain=None, lang=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain = domain
        self.lang = lang

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label,domain,lang):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.domain = domain
        self.lang = lang

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      domain_list=None,
                                      lang_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = processors[task]()
        domain_list=processor.get_domains()
        lang_list=processor.get_langs()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    domain_map= {domain: i for i, domain in enumerate(domain_list)}
    lang_map= {lang: i for i, lang in enumerate(lang_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
            domain=domain_map[example.domain]
            lang=lang_map[example.lang]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)



        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label,
                              domain=domain,
                              lang=lang))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label,
                        ex.domain,
                        ex.lang)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64,tf.int64,tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([])))

    return features, domain_map, lang_map




class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  #def __init__(self,lang):
  #  self.language = lang

  def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "multinli_1.0_train.txt")))

  def set_domains(self, domains):
      self.domains=domains

  def set_languages(self, languages):
      self.languages = languages


  def _create_examples(self, lines):

    examples = []
    langs = set()
    domains = set()
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a =line[5]
      text_b = line[6]
      label = line[0]
      domain=line[9]
      domains.add(domain)
      langs.add('en')

      if label == "contradictory":
        label = "contradiction"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, domain=domain, lang='en'))
    self.set_domains(list(domains))
    self.set_languages(list(langs))
    return examples

  def get_example_from_tensor_dict(self, tensor_dict):
      """See base class."""
      return InputExample(tensor_dict['idx'].numpy(),
                          tensor_dict['premise'].numpy().decode('utf-8'),
                          tensor_dict['hypothesis'].numpy().decode('utf-8'),
                          str(tensor_dict['label'].numpy()),
                          str(tensor_dict['domain'].numpy()),
                          str(tensor_dict['lang'].numpy()))

  def get_dev_examples(self, data_dir, subset):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.{}.tsv".format(subset)))
    examples = []
    langs=set()
    domains=set()
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "subset-%d" % (i)
      language = line[0]
      text_a = line[6]
      text_b = line[7]
      label = line[1]
      domain= line[10]
      domains.add(domain)
      langs.add(language)
      if label == "contradictory":
        label = "contradiction"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, domain=domain, lang=language))

    self.set_domains(list(domains))
    self.set_languages(list(langs))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def get_domains(self):
      return self.domains

  def get_langs(self):
      return self.languages



tasks_num_labels = {

    "mnli": 3,
    "xnli":3
}

processors = {
    "xnli":XnliProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,

}

output_modes = {
    "xnli":"classification",
    "mnli": "classification",
    "mnli-mm": "classification",

}


