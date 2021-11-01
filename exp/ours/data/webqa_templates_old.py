import sys

from allennlp.common import FromParams


class WebQaQueryGenerator(FromParams):
  COMMANDS = "Describe|State|Name".split("|")
  OBJECTS = ["thing", "object", "entity"]

  @staticmethod
  def _replace_noun(source_queries):
    out = []
    for q in source_queries:
      for obj in WebQaQueryGenerator.OBJECTS:
        if obj == "thing" and "the $noun" in q:
          # "the thing" doesn't really sound right
          continue
        out.append(q.replace("$noun", obj))
    return out

  def __init__(self, version=0):
    self.version = version

    # Extra sample of the main query
    a2_queries = ['What $adj_type is this $noun?']
    for prefix in ["this", "the", "that"]:
      for qprefix in ["What", "Which"]:
        a2_queries += [f"{qprefix} $adj_type is {prefix} $noun?"]
      a2_queries += [f"What is the $adj_type of {prefix} $noun?"]
      for cmd in self.COMMANDS:
        a2_queries.append(f"{cmd} the $adj_type of {prefix} $noun.")

    a1_queries = self._replace_noun(a2_queries)

    v2_queries = []
    for prefix in ["this", "the", "that"]:
      # Oversample the main query
      v2_queries += [f"What is {prefix} $noun doing?"]*4

      v2_queries.append(f"What action is {prefix} $noun taking?")
      v2_queries.append(f"What action is {prefix} $noun doing?")
      v2_queries.append(f"What activity is {prefix} $noun doing?")

      for cmd in self.COMMANDS:
        v2_queries.append(f"{cmd} the action being taken by {prefix} $noun.")
        v2_queries.append(f"{cmd} the activity {prefix} $noun is doing.")
        v2_queries.append(f"{cmd} what {prefix} $noun is doing.")

    v1_queries = []
    for q in ["What", "Which"]:
      v1_queries += [
                      f"{q} action is being done?",
                      f"{q} activity is being done?",
                      f"{q} activity is this?",
                      f"{q} action is being taken?"
                    ] * 2
    v1_queries += ["What is being done?"] * 3
    v1_queries += self._replace_noun(v2_queries)

    n1_queries = []
    for prefix in ["this", "that"]:
      n1_queries += [f"What object is {prefix}?"]*3
      n1_queries += [f"What is {prefix}?"]*2
      n1_queries += [f"What entity is {prefix}?"]*2

      for obj in self.OBJECTS:
        n1_queries += [f"What is {prefix} {obj}?"]
        for cmd in ["Name", "Describe", "Classify"]:
          n1_queries.append(f"{cmd} {prefix} {obj}.")

    self.train_types = {
      "1n": n1_queries,
      "1v": v1_queries,
      "1a": a1_queries,
      "2a": a2_queries,
      "2v": v2_queries,
    }

    self.test_types = {k: v[:1] for k, v  in self.train_types.items()}

  def get_prompts(self, x, is_train=True):
    if is_train:
      templates = self.train_types[x.qtype]
    else:
      templates = self.test_types[x.qtype]

    if x.question_type in {"1n", "1v"}:
      return templates

    if x.question_type == "2v":
      assert x.query.startswith("What is this ")
      assert x.query.endswith(" doing?")
      noun = x.query[len("What is this "):-len(" doing?")]
      return [sys.intern(x.replace("$noun", noun)) for x in templates]
    elif x.question_type == "2a":
      assert x.query.startswith("What ")
      assert x.query.endswith("?")
      q = x.query[len("What "):-1]
      adj_type, noun = q.split(" is this ")
      return [sys.intern(x.replace("$noun", noun).replace("$adj_type", adj_type)) for x in templates]
    elif x.question_type == "1a":
      assert x.query.startswith("What ")
      assert x.query.endswith(" is this entity?")
      adj_type = x.query[len("What "):-len(" is this entity?")]
      return [sys.intern(x.replace("$adj_type", adj_type)) for x in templates]
    else:
      raise NotImplementedError()
