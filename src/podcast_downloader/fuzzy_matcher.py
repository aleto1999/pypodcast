"""advanced fuzzy matching for podcast discovery.

provides multi-algorithm matching for more accurate podcast search results:
- word-based matching with position scoring
- character-based similarity
- phonetic matching (soundex)
- podcast-specific abbreviation handling
- query variation generation
"""

import re
from typing import NamedTuple


class MatchResult(NamedTuple):
    """result of fuzzy matching."""

    score: float  # 0.0 to 1.0
    explanation: str  # human-readable explanation


class MatchConfig(NamedTuple):
    """configuration for fuzzy matching."""

    word_weight: float = 0.4
    char_weight: float = 0.3
    phonetic_weight: float = 0.2
    position_weight: float = 0.1
    min_word_length: int = 2
    use_abbreviations: bool = True


# podcast-specific abbreviations.
ABBREVIATIONS = {
    "npr": "national public radio",
    "bbc": "british broadcasting corporation",
    "nyt": "new york times",
    "wsj": "wall street journal",
    "pbs": "public broadcasting service",
    "abc": "american broadcasting company",
    "cbs": "columbia broadcasting system",
    "nbc": "national broadcasting company",
    "cnn": "cable news network",
    "msnbc": "microsoft national broadcasting company",
    "pod": "podcast",
    "ep": "episode",
    "pts": "parts",
    "pt": "part",
    "ft": "featuring",
    "w/": "with",
    "vs": "versus",
    "&": "and",
}

# stopwords to filter from queries.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "that",
    "this", "these", "those", "it", "its", "podcast", "show", "episode",
}


def soundex(word: str) -> str:
    """compute soundex code for a word.

    soundex is a phonetic algorithm that encodes words based on how
    they sound, useful for matching words with different spellings.

    args:
        word: word to encode.

    returns:
        4-character soundex code.
    """
    if not word:
        return "0000"

    word = word.upper()
    word = re.sub(r"[^A-Z]", "", word)

    if not word:
        return "0000"

    # soundex code mapping.
    code_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    # first letter stays as is.
    result = word[0]

    # encode remaining letters.
    prev_code = code_map.get(word[0], "0")
    for char in word[1:]:
        code = code_map.get(char, "0")
        if code != "0" and code != prev_code:
            result += code
            prev_code = code

    # pad to 4 characters.
    result = (result + "000")[:4]
    return result


def normalize_text(text: str, expand_abbreviations: bool = True) -> str:
    """normalize text for matching.

    args:
        text: text to normalize.
        expand_abbreviations: whether to expand abbreviations.

    returns:
        normalized text.
    """
    text = text.lower()

    # expand abbreviations.
    if expand_abbreviations:
        for abbr, expansion in ABBREVIATIONS.items():
            # only expand if abbreviation is a word boundary.
            pattern = rf"\b{re.escape(abbr)}\b"
            text = re.sub(pattern, expansion, text)

    # remove special characters.
    text = re.sub(r"[^\w\s]", " ", text)

    # collapse whitespace.
    text = " ".join(text.split())

    return text


def extract_words(text: str, min_length: int = 2) -> list[str]:
    """extract significant words from text.

    args:
        text: text to extract words from.
        min_length: minimum word length to include.

    returns:
        list of significant words.
    """
    words = normalize_text(text).split()
    return [w for w in words if len(w) >= min_length and w not in STOPWORDS]


def word_match_score(query_words: list[str], target_words: list[str]) -> float:
    """calculate word-based match score.

    args:
        query_words: words from the query.
        target_words: words from the target.

    returns:
        score from 0.0 to 1.0.
    """
    if not query_words or not target_words:
        return 0.0

    matches = sum(1 for word in query_words if word in target_words)
    return matches / len(query_words)


def word_position_score(query_words: list[str], target_words: list[str]) -> float:
    """calculate position-aware word match score.

    words appearing earlier in both query and target get higher scores.

    args:
        query_words: words from the query.
        target_words: words from the target.

    returns:
        score from 0.0 to 1.0.
    """
    if not query_words or not target_words:
        return 0.0

    score = 0.0
    total_weight = 0.0

    for i, query_word in enumerate(query_words):
        # weight decreases with position.
        weight = 1.0 / (i + 1)
        total_weight += weight

        for j, target_word in enumerate(target_words):
            if query_word == target_word:
                # bonus for matching early positions.
                position_bonus = 1.0 / (abs(i - j) + 1)
                score += weight * position_bonus
                break

    return score / total_weight if total_weight > 0 else 0.0


def levenshtein_ratio(s1: str, s2: str) -> float:
    """calculate levenshtein similarity ratio.

    args:
        s1: first string.
        s2: second string.

    returns:
        similarity ratio from 0.0 to 1.0.
    """
    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)

    # use dynamic programming for edit distance.
    prev_row = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr_row = [i]
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                curr_row.append(prev_row[j - 1])
            else:
                curr_row.append(1 + min(prev_row[j - 1], prev_row[j], curr_row[j - 1]))
        prev_row = curr_row

    distance = prev_row[len2]
    return 1.0 - (distance / max_len)


def phonetic_match_score(query_words: list[str], target_words: list[str]) -> float:
    """calculate phonetic similarity score using soundex.

    args:
        query_words: words from the query.
        target_words: words from the target.

    returns:
        score from 0.0 to 1.0.
    """
    if not query_words or not target_words:
        return 0.0

    query_codes = [soundex(w) for w in query_words]
    target_codes = [soundex(w) for w in target_words]

    matches = sum(1 for code in query_codes if code in target_codes)
    return matches / len(query_codes)


def fuzzy_match(
    query: str,
    target: str,
    config: MatchConfig | None = None,
) -> MatchResult:
    """perform fuzzy matching between query and target.

    combines multiple algorithms for robust matching:
    - word-based matching (0.4 weight)
    - character-based similarity (0.3 weight)
    - phonetic matching (0.2 weight)
    - position scoring (0.1 weight)

    args:
        query: the search query.
        target: the target string to match against.
        config: matching configuration.

    returns:
        match result with score and explanation.
    """
    config = config or MatchConfig()

    # extract words.
    query_words = extract_words(query, config.min_word_length)
    target_words = extract_words(target, config.min_word_length)

    if not query_words:
        return MatchResult(score=0.0, explanation="no significant words in query")

    if not target_words:
        return MatchResult(score=0.0, explanation="no significant words in target")

    # calculate individual scores.
    word_score = word_match_score(query_words, target_words)
    char_score = levenshtein_ratio(normalize_text(query), normalize_text(target))
    phonetic_score = phonetic_match_score(query_words, target_words)
    position_score = word_position_score(query_words, target_words)

    # weighted combination.
    total_score = (
        word_score * config.word_weight
        + char_score * config.char_weight
        + phonetic_score * config.phonetic_weight
        + position_score * config.position_weight
    )

    # build explanation.
    parts = []
    if word_score > 0.3:
        matching_words = [w for w in query_words if w in target_words]
        parts.append(f"matched words: {', '.join(matching_words)}")
    if phonetic_score > word_score:
        parts.append("phonetic similarity detected")
    if position_score > 0.5:
        parts.append("good word ordering")

    explanation = "; ".join(parts) if parts else "low match confidence"

    return MatchResult(score=min(total_score, 1.0), explanation=explanation)


def generate_query_variations(query: str) -> list[str]:
    """generate variations of a search query.

    creates variations that might improve search results:
    - original query
    - with "podcast" added
    - with "podcast" removed
    - with "the" removed
    - with abbreviations expanded

    args:
        query: the original search query.

    returns:
        list of query variations (including original).
    """
    variations = [query]

    # normalized version.
    normalized = normalize_text(query, expand_abbreviations=False)
    if normalized != query.lower():
        variations.append(normalized)

    # with abbreviations expanded.
    expanded = normalize_text(query, expand_abbreviations=True)
    if expanded not in variations:
        variations.append(expanded)

    # add "podcast" suffix.
    if "podcast" not in query.lower():
        variations.append(f"{query} podcast")

    # remove "podcast" suffix.
    if "podcast" in query.lower():
        no_podcast = re.sub(r"\s*podcast\s*", " ", query, flags=re.IGNORECASE).strip()
        if no_podcast and no_podcast not in variations:
            variations.append(no_podcast)

    # remove leading "the".
    if query.lower().startswith("the "):
        no_the = query[4:]
        if no_the and no_the not in variations:
            variations.append(no_the)

    # add leading "the" for common patterns.
    words = query.split()
    if words and words[0].lower() not in ["the", "a", "an"]:
        with_the = f"the {query}"
        if with_the not in variations:
            variations.append(with_the)

    return variations


def rank_results(
    query: str,
    results: list[tuple[str, any]],
    title_extractor: callable = lambda x: x[0],
) -> list[tuple[str, any, MatchResult]]:
    """rank search results by fuzzy match score.

    args:
        query: the search query.
        results: list of (title, data) tuples to rank.
        title_extractor: function to extract title from result.

    returns:
        list of (title, data, match_result) tuples sorted by score.
    """
    scored_results = []

    for result in results:
        title = title_extractor(result)
        match = fuzzy_match(query, title)
        scored_results.append((result[0], result[1], match))

    # sort by score descending.
    scored_results.sort(key=lambda x: x[2].score, reverse=True)

    return scored_results


def find_best_match(
    query: str,
    candidates: list[str],
    min_score: float = 0.3,
) -> tuple[str, MatchResult] | None:
    """find the best matching candidate for a query.

    args:
        query: the search query.
        candidates: list of candidate strings.
        min_score: minimum score threshold.

    returns:
        tuple of (best_candidate, match_result) or none if no match.
    """
    best_candidate = None
    best_result = None

    for candidate in candidates:
        result = fuzzy_match(query, candidate)
        if result.score >= min_score:
            if best_result is None or result.score > best_result.score:
                best_candidate = candidate
                best_result = result

    if best_candidate:
        return (best_candidate, best_result)
    return None
