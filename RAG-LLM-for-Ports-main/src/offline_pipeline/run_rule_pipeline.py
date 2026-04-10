from pattern_detector import detect_patterns
from rule_extractor import extract_rules
from rule_normalizer import normalize

def run():

    print("Stage 5 Pattern detection")
    detect_patterns()

    print("Stage 6 LLM rule extraction")
    extract_rules()

    print("Stage 7 Rule normalization")
    normalize()

if __name__ == "__main__":
    run()