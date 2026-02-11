"""Politeness feature extraction for podcast conversations."""

from dataclasses import dataclass, field

# politeness strategy weights from convokit/stanford politeness.
POLITENESS_WEIGHTS = {
    "Bias": -0.219798,
    # politeness markers.
    "Gratitude": 2.11927,
    "Deference": 1.22837,
    "Greeting": 0.77941,
    "Positive": 0.68555,
    "Negative": -1.14790,
    "Please": 1.31409,
    "Please_start": -1.71990,
    "Indirect_(btw)": 1.18813,
    "Hedges": 0.61410,
    "Factuality": -1.04980,
    "Apologizing": 1.50141,
    # person markers.
    "1st_person": 0.27582,
    "1st_person_pl.": 0.21695,
    "1st_person_start": 0.01042,
    "2nd_person": -0.04706,
    "2nd_person_start": -0.43240,
    # directness.
    "Direct_question": -1.18020,
    "Direct_start": -1.54560,
    # modality.
    "Counterfactual_modal": 0.94153,
    "Indicative_modal": 0.35432,
}

# mapping from convokit strategy names to weight keys.
FEATURE_MAP = {
    "==Gratitude==": "Gratitude",
    "==Deference==": "Deference",
    "==Indirect_(greeting)==": "Greeting",
    "==Positive_Lexicon==": "Positive",
    "==Negative_Lexicon==": "Negative",
    "==Please==": "Please",
    "==Please_start==": "Please_start",
    "==Indirect_(btw)==": "Indirect_(btw)",
    "==Hedges==": "Hedges",
    "==Factuality==": "Factuality",
    "==Apologizing==": "Apologizing",
    "==1st_person==": "1st_person",
    "==1st_person_pl==": "1st_person_pl.",
    "==1st_person_start==": "1st_person_start",
    "==2nd_person==": "2nd_person",
    "==2nd_person_start==": "2nd_person_start",
    "==Direct_question==": "Direct_question",
    "==Direct_start==": "Direct_start",
    "==Subjunctive==": "Counterfactual_modal",
    "==Indicative==": "Indicative_modal",
}


@dataclass
class PolitenessStats:
    """Statistics about politeness in a conversation."""

    average_politeness_score: float = 0.0
    politeness_by_speaker: dict = field(default_factory=dict)
    strategy_counts: dict = field(default_factory=dict)
    total_sentences_analyzed: int = 0


class PolitenessAnalyzer:
    """Analyzes politeness features in podcast transcripts.

    Note: For full functionality, requires convokit and spacy.
    Falls back to simple heuristic-based analysis if not available.
    """

    def __init__(self, use_convokit: bool = True):
        """Initialize the politeness analyzer.

        Args:
            use_convokit: Whether to use convokit for analysis.
                         Falls back to heuristics if False or if import fails.
        """
        self.use_convokit = use_convokit
        self.ps = None
        self.nlp = None

        if use_convokit:
            try:
                from convokit import PolitenessStrategies
                import spacy

                self.ps = PolitenessStrategies()
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # spacy model not installed.
                    self.nlp = None
                    self.use_convokit = False
            except ImportError:
                self.use_convokit = False

    def calculate_politeness_score(self, strategies: dict) -> float:
        """Calculate weighted politeness score from strategy counts."""
        score = POLITENESS_WEIGHTS.get("Bias", 0.0)

        for strategy_key, weight_key in FEATURE_MAP.items():
            if strategy_key in strategies and strategies[strategy_key]:
                weight = POLITENESS_WEIGHTS.get(weight_key, 0.0)
                # strategies can be bool or count.
                count = strategies[strategy_key]
                if isinstance(count, bool):
                    count = 1 if count else 0
                elif isinstance(count, list):
                    count = len(count)
                score += weight * count

        return score

    def analyze_sentence_convokit(self, sentence: str) -> tuple[float, dict]:
        """Analyze a single sentence using convokit."""
        if not self.ps or not self.nlp:
            return 0.0, {}

        try:
            utt = self.ps.transform_utterance(sentence, spacy_nlp=self.nlp)
            strategies = utt.meta.get("politeness_strategies", {})
            score = self.calculate_politeness_score(strategies)
            return score, strategies
        except Exception:
            return 0.0, {}

    def analyze_sentence_heuristic(self, sentence: str) -> tuple[float, dict]:
        """Analyze a single sentence using simple heuristics."""
        sentence_lower = sentence.lower()
        strategies = {}
        score = POLITENESS_WEIGHTS.get("Bias", 0.0)

        # gratitude.
        if any(w in sentence_lower for w in ["thank", "thanks", "appreciate"]):
            strategies["Gratitude"] = True
            score += POLITENESS_WEIGHTS.get("Gratitude", 0.0)

        # please.
        if "please" in sentence_lower:
            strategies["Please"] = True
            score += POLITENESS_WEIGHTS.get("Please", 0.0)
            if sentence_lower.strip().startswith("please"):
                strategies["Please_start"] = True
                score += POLITENESS_WEIGHTS.get("Please_start", 0.0)

        # apologizing.
        if any(w in sentence_lower for w in ["sorry", "apologize", "apologies"]):
            strategies["Apologizing"] = True
            score += POLITENESS_WEIGHTS.get("Apologizing", 0.0)

        # hedges.
        hedges = ["maybe", "perhaps", "possibly", "might", "could", "i think", "i guess"]
        if any(h in sentence_lower for h in hedges):
            strategies["Hedges"] = True
            score += POLITENESS_WEIGHTS.get("Hedges", 0.0)

        # direct question.
        if sentence.strip().endswith("?"):
            strategies["Direct_question"] = True
            score += POLITENESS_WEIGHTS.get("Direct_question", 0.0)

        # second person.
        if any(w in sentence_lower.split() for w in ["you", "your", "yours"]):
            strategies["2nd_person"] = True
            score += POLITENESS_WEIGHTS.get("2nd_person", 0.0)

        # first person.
        if any(w in sentence_lower.split() for w in ["i", "my", "me", "mine"]):
            strategies["1st_person"] = True
            score += POLITENESS_WEIGHTS.get("1st_person", 0.0)

        # first person plural.
        if any(w in sentence_lower.split() for w in ["we", "our", "us", "ours"]):
            strategies["1st_person_pl."] = True
            score += POLITENESS_WEIGHTS.get("1st_person_pl.", 0.0)

        return score, strategies

    def analyze(self, segments: list[dict], nlp=None) -> PolitenessStats:
        """Analyze politeness from transcript segments.

        Args:
            segments: List of transcript segments with 'text' and 'speaker' keys.
            nlp: Optional spacy nlp model for sentence splitting.

        Returns:
            PolitenessStats with politeness analysis results.
        """
        stats = PolitenessStats()
        scores_by_speaker: dict[str, list[float]] = {}
        all_strategy_counts: dict[str, int] = {}

        for seg in segments:
            text = seg.get("text", "")
            speaker = seg.get("speaker", "UNKNOWN")

            if not text.strip():
                continue

            # analyze the segment.
            if self.use_convokit:
                score, strategies = self.analyze_sentence_convokit(text)
            else:
                score, strategies = self.analyze_sentence_heuristic(text)

            stats.total_sentences_analyzed += 1

            # accumulate by speaker.
            if speaker not in scores_by_speaker:
                scores_by_speaker[speaker] = []
            scores_by_speaker[speaker].append(score)

            # count strategies.
            for strategy, value in strategies.items():
                if value:
                    if strategy not in all_strategy_counts:
                        all_strategy_counts[strategy] = 0
                    all_strategy_counts[strategy] += 1

        # calculate averages.
        all_scores = []
        for speaker, scores in scores_by_speaker.items():
            if scores:
                avg = sum(scores) / len(scores)
                stats.politeness_by_speaker[speaker] = avg
                all_scores.extend(scores)

        if all_scores:
            stats.average_politeness_score = sum(all_scores) / len(all_scores)

        stats.strategy_counts = all_strategy_counts

        return stats

    def to_dict(self, stats: PolitenessStats) -> dict:
        """Convert PolitenessStats to dictionary for JSON serialization."""
        return {
            "average_politeness_score": round(stats.average_politeness_score, 4),
            "total_sentences_analyzed": stats.total_sentences_analyzed,
            "politeness_by_speaker": {
                speaker: round(score, 4)
                for speaker, score in stats.politeness_by_speaker.items()
            },
            "strategy_counts": stats.strategy_counts,
        }

    def to_flat_dict(self, stats: PolitenessStats) -> dict:
        """Convert PolitenessStats to flat dictionary for CSV export."""
        flat = {
            "average_politeness_score": round(stats.average_politeness_score, 4),
            "total_sentences_analyzed": stats.total_sentences_analyzed,
        }

        # flatten speaker politeness.
        for speaker, score in stats.politeness_by_speaker.items():
            flat[f"politeness_{speaker}"] = round(score, 4)

        # flatten strategy counts.
        for strategy, count in stats.strategy_counts.items():
            flat[f"strategy_{strategy}"] = count

        return flat
