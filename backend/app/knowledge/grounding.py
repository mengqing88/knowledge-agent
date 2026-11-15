"""Conservative numeric and coverage guards, independent of the Verifier model.

This checks literal bindings and arithmetic, not the truth of a source or arbitrary
natural-language entailment. Ambiguous tables, unsupported units and formulas are
withheld for revision instead of silently receiving a numeric pass.
"""
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext

NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")
UNSUPPORTED_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?[eE][+-]?\d+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"(?<![A-Za-z0-9_个])[负零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?\s*"
    r"(?:个百分点|毫秒|秒|字节|分|个|次|GB|MB|ms)(?![率点之比])"
)
CLAUSE = re.compile(r"\band\b|\bbut\b|\bwhereas\b|\bwhile\b|[;；。\n]|\.(?!\d)", re.I)
UNITS = {
    "number": ("number", Decimal(1), ("number", "score", "分", "个")),
    "%": ("ratio", Decimal(".01"), ("%", "％", "percent", "百分比")),
    "pp": ("ratio_delta", Decimal(".01"), ("percentage points", "percentage point", "pp", "个百分点")),
    "ms": ("time", Decimal(".001"), ("milliseconds", "millisecond", "ms", "毫秒")),
    "s": ("time", Decimal(1), ("seconds", "second", "sec", "s", "秒")),
    "us": ("time", Decimal(".000001"), ("microseconds", "us", "μs", "微秒")),
    "B": ("memory", Decimal(1), ("bytes", "byte", "B", "字节")),
    "MB": ("memory", Decimal(1000000), ("MB",)),
    "GB": ("memory", Decimal(1000000000), ("GB",)),
    "MiB": ("memory", Decimal(1048576), ("MiB",)),
    "GiB": ("memory", Decimal(1073741824), ("GiB",)),
}
METRICS = (
    ("f1 score", "f1", "f1分数", "f1得分"),
    ("accuracy", "准确率", "正确率"),
    ("average inference latency", "inference latency", "average latency", "latency",
     "平均推理延迟", "推理延迟", "平均延迟", "延迟"),
    ("gpu memory", "memory", "显存", "内存"),
)


def normalized(value):
    return re.sub(r"\s+", "", value).casefold()


def contains(text, label):
    # Identifier boundaries keep M from matching measured/method and F1 from F10.
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(label.strip()) +
                          r"(?![A-Za-z0-9_])", text, re.I))


def metric_present(text, metric):
    aliases = next((group for group in METRICS if normalized(metric) in map(normalized, group)), (metric,))
    return any(contains(text, alias) for alias in aliases)


def condition_present(text, condition):
    if not condition.strip():
        return False
    condition = condition.strip()
    # Ignore spacing inside a condition but do not match part of an ASCII
    # identifier: "setting A" must not be supported by "setting AB".
    parts = re.split(r"\s+", condition)
    pattern = r"\s*".join(re.escape(part) for part in parts)
    if re.match(r"[A-Za-z0-9_]", condition):
        pattern = r"(?<![A-Za-z0-9_])(?<![A-Za-z0-9_][.-])" + pattern
    if re.search(r"[A-Za-z0-9_]$", condition):
        pattern += r"(?![A-Za-z0-9_]|[.-][A-Za-z0-9_])"
    if re.search(pattern, text, re.I):
        return True
    dataset = re.fullmatch(r"(?:dataset|数据集)\s*([A-Za-z0-9_-]+)", condition.strip(), re.I)
    return bool(dataset and normalized(dataset[1]) in dataset_labels(text))


def canonical_unit(label):
    return next((key for key, (_, _, aliases) in UNITS.items()
                 if label.casefold().strip() in (a.casefold() for a in aliases)), None)


def adjacent_unit(text, match):
    tail = text[match.end():].lstrip()
    aliases = sorted(((alias, key) for key, (_, _, values) in UNITS.items()
                      if key != "number" for alias in values), key=lambda pair: -len(pair[0]))
    for alias, key in aliases:
        if tail.casefold().startswith(alias.casefold()):
            rest = tail[len(alias):]
            if alias[-1].isascii() and alias[-1].isalpha() and rest and rest[0].isascii() and rest[0].isalpha():
                continue
            return key
    return "number"


def quantities(text):
    return [(Decimal(m.group()), adjacent_unit(text, m), m) for m in NUMBER.finditer(text)]


def local_clause(text, match):
    start, end = 0, len(text)
    for boundary in CLAUSE.finditer(text):
        if boundary.group().casefold() == "and" and not NUMBER.search(text[start:boundary.start()]):
            continue
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.end():
            end = boundary.start()
            break
    return text[start:end]


def fact_value(fact, sources, support_ids):
    source = sources.get(fact["evidence_id"])
    if not source or fact["evidence_id"] not in support_ids:
        raise ValueError("数值只能绑定本论点的 supports 证据")
    quote, full = fact["quote"], source["quote"]
    if quote not in full:
        raise ValueError("数值定位不是证据的逐字摘录")
    unit = canonical_unit(fact["unit"])
    if unit is None:
        raise ValueError("数值单位尚不支持程序校验")
    value = Decimal(fact["value"])
    quote_content = quote.strip().rstrip(".。;；\r\n")
    if not quote_content or not any(number == value for number, _, _ in quantities(quote)):
        raise ValueError("数值摘录必须包含所绑定的原值")
    # Locate in the full supplied span, so trimming '%' off the small quote cannot
    # relabel a percentage as a unitless score. Every occurrence must be unambiguous.
    bindings = []
    for number, actual_unit, match in quantities(full):
        if number != value or actual_unit != unit:
            continue
        clause = local_clause(full, match)
        if (quote_content not in clause or not contains(clause, fact["subject"]) or
                not metric_present(clause, fact["metric"])):
            continue
        if len(quantities(clause)) != 1:
            continue
        if not all(condition_present(clause, condition) or (
                re.fullmatch(r"(?:dataset|数据集)\s*[A-Za-z0-9_-]+", condition.strip(), re.I) and
                len(dataset_labels(full)) == 1 and condition_present(full, condition))
                for condition in fact["conditions"]):
            continue
        bindings.append((number * UNITS[unit][1], UNITS[unit][0]))
    if len(bindings) != 1:
        raise ValueError("原文中的对象、指标、数值或单位无法唯一配对；需提供更清晰的原文")
    return bindings[0]


def comparable(facts, sources):
    conditions = [{normalized(c) for c in f["conditions"]} for f in facts]
    if not conditions[0] or conditions[0] != conditions[1]:
        raise ValueError("比较必须逐项绑定相同且非空的实验条件")
    # Independently catch a common cross-dataset mistake even if the model omitted
    # dataset names from its structured conditions. Other settings remain semantic review.
    datasets = []
    for fact in facts:
        quote = sources[fact["evidence_id"]]["quote"]
        names = {normalized(m) for m in re.findall(r"(?:dataset|数据集)\s*([A-Za-z0-9_-]+)", quote, re.I)}
        datasets.append(names)
    if datasets[0] != datasets[1] or any(len(names) > 1 for names in datasets):
        raise ValueError("原文的数据集条件不同或不明确，不能直接比较")
    first_metric = next((i for i, group in enumerate(METRICS)
                         if normalized(facts[0]["metric"]) in map(normalized, group)), None)
    second_metric = next((i for i, group in enumerate(METRICS)
                          if normalized(facts[1]["metric"]) in map(normalized, group)), None)
    if (first_metric != second_metric or
            (first_metric is None and normalized(facts[0]["metric"]) != normalized(facts[1]["metric"]))):
        raise ValueError("不同指标不能直接比较")


def dataset_labels(text):
    return {normalized(m) for m in re.findall(r"(?:dataset|数据集)\s*([A-Za-z0-9_-]+)", text, re.I)}


def operand_bound(statement, fact):
    # Derived statements keep each raw value next to its own subject and metric;
    # the presence of both entities and both numbers anywhere is not sufficient.
    parts = re.split(r"\band\b|\bversus\b|\bvs\.?\b|[与和对，,；;。\n]|\.(?!\d)", statement, flags=re.I)
    return any(len(quantities(part)) == 1 and contains(part, fact["subject"]) and
               metric_present(part, fact["metric"]) and any(
                   value == Decimal(fact["value"]) and unit == canonical_unit(fact["unit"])
                   for value, unit, _ in quantities(part)) for part in parts)


def check_numeric(check, claim, sources, support_ids):
    statement = check["statement"]
    if statement not in claim["text"]:
        raise ValueError("数值检查的 statement 必须逐字出现在论点中")
    unit = canonical_unit(check["unit"])
    if unit is None:
        raise ValueError("输出单位尚不支持程序校验")
    facts = check["facts"]
    values = [fact_value(fact, sources, support_ids) for fact in facts]
    for fact in facts:
        if not contains(statement, fact["subject"]) or not metric_present(statement, fact["metric"]):
            raise ValueError("论点中的对象或指标未与原文数值绑定；对象保留原文名称")
        declared_text = " ".join([statement, *claim.get("conditions", [])])
        if any(not condition_present(declared_text, condition) for condition in fact["conditions"]):
            raise ValueError("论点必须保留数值所适用的原文条件")
        declared = dataset_labels(declared_text)
        original = dataset_labels(sources[fact["evidence_id"]]["quote"])
        if declared and original and declared != original:
            raise ValueError("论点中的数据集条件与数值原文不一致")
    operation = check["operation"]
    dimension, scale, _ = UNITS[unit]
    if operation == "reported":
        if len(values) != 1 or values[0][1] != dimension or len(quantities(statement)) != 1:
            raise ValueError("原值与输出的数量或单位类型不符")
        expected = values[0][0] / scale
    else:
        if len(values) != 2 or values[0][1] != values[1][1]:
            raise ValueError("比较需要两个同类指标值")
        comparable(facts, sources)
        if operation == "difference":
            expected_dim = "ratio_delta" if values[0][1] == "ratio" else values[0][1]
            if dimension != expected_dim:
                raise ValueError("差值单位不符：百分比之差必须使用百分点")
            expected = (values[0][0] - values[1][0]) / scale
        else:
            if unit != "%" or not values[1][0]:
                raise ValueError("相对变化必须用百分比，且基准值不能为零")
            expected = (values[0][0] - values[1][0]) / abs(values[1][0]) * 100
        # Keep operands visible to a reader; server determines arithmetic, never eval().
        if any(not operand_bound(statement, f) for f in facts):
            raise ValueError("推导论点必须逐个绑定对象、指标和原值，并显示差值/相对变化关系")
        marker = r"差|相减|减去|difference|minus|−| - " if operation == "difference" else r"相对|relative|%|％"
        if not re.search(marker, statement, re.I):
            raise ValueError("推导论点缺少运算关系")
    actual = Decimal(check["value"])
    rounded = (expected.quantize(Decimal(1).scaleb(actual.as_tuple().exponent), rounding=ROUND_HALF_UP)
               if operation == "relative_change" else expected)
    if rounded != actual:
        raise ValueError("数值与单位换算或程序计算结果不一致")
    matches = [(m.start(), m.end()) for value, actual_unit, m in quantities(statement)
               if value == actual and actual_unit == unit]
    if not matches:
        raise ValueError("声明的结果数值及单位没有逐字出现在论点中")
    return statement


def numeric_issues(claim, sources):
    issues, checked_statements, checked_conditions = [], [], set()
    if UNSUPPORTED_NUMBER.search(" ".join([claim["text"], *claim.get("conditions", [])])):
        issues.append({"claim_id": claim["id"], "type": "numeric_mismatch",
                       "message": "该数值记法尚不支持程序核对，请改用明确十进制数值或标记缺口"})
    support_ids = {e["evidence_id"] for e in claim["evidence"] if e["relation"] == "supports"}
    for check in claim.get("numeric_checks", []):
        try:
            with localcontext() as context:
                context.prec = 80
                statement = check_numeric(check, claim, sources, support_ids)
            # A check covers its declared result and bound operands only, not every
            # number in an arbitrarily large statement.
            allowed = {(Decimal(check["value"]), canonical_unit(check["unit"]))}
            allowed.update((Decimal(f["value"]), canonical_unit(f["unit"])) for f in check["facts"])
            checked_statements.append((statement, allowed))
            checked_conditions.update(normalized(condition) for fact in check["facts"]
                                      for condition in fact["conditions"])
        except (ValueError, InvalidOperation, ZeroDivisionError) as exc:
            issues.append({"claim_id": claim["id"], "type": "comparison_mismatch" if
                           check["operation"] != "reported" else "numeric_mismatch", "message": str(exc)})
    for value, unit, match in quantities(claim["text"]):
        if not any((value, unit) in allowed and any(
                occurrence.start() <= match.start() and match.end() <= occurrence.end()
                for occurrence in re.finditer(re.escape(statement), claim["text"]))
                for statement, allowed in checked_statements):
            issues.append({"claim_id": claim["id"], "type": "numeric_mismatch",
                           "message": f"数值 {value}（{unit}）缺少已通过的原文绑定和程序校验"})
    for condition in claim.get("conditions", []):
        for value, unit, _ in quantities(condition):
            if (normalized(condition) not in checked_conditions or
                    not any((value, unit) in allowed for _, allowed in checked_statements)):
                issues.append({"claim_id": claim["id"], "type": "numeric_mismatch",
                               "message": "条件中的数值没有绑定到已校验原文"})
    return issues


def coverage_issues(plan, draft):
    expected = {q["id"] for q in plan.get("subquestions", [])}
    issues = []
    for claim in draft["claims"]:
        labels = claim.get("subquestion_ids", [])
        if not labels or not set(labels) <= expected:
            issues.append({"claim_id": claim["id"], "type": "unsupported_claim",
                           "message": "论点必须关联本次计划中有效的子问题 ID"})
    for gap in draft.get("subquestion_gaps", []):
        if gap["subquestion_id"] not in expected:
            issues.append({"claim_id": None, "type": "missing_evidence", "message": "缺口引用了未知子问题 ID"})
    return issues


def uncovered_questions(plan, draft, claims):
    expected = {q["id"] for q in plan.get("subquestions", [])}
    covered = {qid for claim in claims for qid in claim.get("subquestion_ids", []) if qid in expected}
    reasons = {gap["subquestion_id"]: gap["reason"] for gap in draft.get("subquestion_gaps", [])
               if gap["subquestion_id"] in expected}
    return [{"subquestion_id": q["id"], "target": q["target"],
             "reason": reasons.get(q["id"], "当前材料及本次检索未形成通过检查的回答。")}
            for q in plan.get("subquestions", []) if q["id"] not in covered or q["id"] in reasons]
