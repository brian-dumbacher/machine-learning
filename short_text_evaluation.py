
import io
import math
from nltk.classify import *
from nltk.classify.util import *
#from nltk.metrics import *
import random
import re
#from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import *
#from sklearn.model_selection import *
#from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.tree import *
import statistics

#################################################
###   Text cleaning variables and functions   ###
#################################################

__stop_words = (
    "a",
    "about",
    "again",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "b",
    "be",
    "because",
    "been",
    "being",
    "both",
    "but",
    "by",
    "c",
    "d",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "during",
    "e",
    "each",
    "f",
    "for",
    "from",
    "g",
    "h",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "j",
    "just",
    "k",
    "l",
    "m",
    "me",
    "mine",
    "my",
    "myself",
    "n",
    "o",
    "of",
    "off",
    "on",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "p",
    "q",
    "r",
    "s",
    "she",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "u",
    "us",
    "v",
    "w",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
    "x",
    "y",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "z",
)
__vowels = "aeiouy"
__double_consonants = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")
__li_ending = "cdeghkmnrt"
__step1a_suffixes = ("sses", "ied", "ies", "us", "ss", "s")
__step1b_suffixes = ("eedly", "ingly", "edly", "eed", "ing", "ed")
__step2_suffixes = (
    "ization",
    "ational",
    "fulness",
    "ousness",
    "iveness",
    "tional",
    "biliti",
    "lessli",
    "entli",
    "ation",
    "alism",
    "aliti",
    "ousli",
    "iviti",
    "fulli",
    "enci",
    "anci",
    "abli",
    "izer",
    "ator",
    "alli",
    "bli",
    "ogi",
    "li",
)
__step3_suffixes = (
    "ational",
    "tional",
    "alize",
    "icate",
    "iciti",
    "ative",
    "ical",
    "ness",
    "ful",
)
__step4_suffixes = (
    "ement",
    "ance",
    "ence",
    "able",
    "ible",
    "ment",
    "ant",
    "ent",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
    "ion",
    "al",
    "er",
    "ic",
)
#__step5_suffixes = ("e", "l")
__step6_suffixes = (
    "curist",
    "graphi",
    "logi",
    "logist",
    "nomi",
    "nomist",
    "pathi",
    "pathet",
    "physicist",
    "scopi",
    "therapeut",
    "therapi",
    "therapist",
    "tomi",
    "tomist",
    "tri",
    "trist",
    "trician",
    "turist",
)
__special_words = {
    "skis": "ski",
    "skies": "sky",
    "dying": "die",
    "lying": "lie",
    "tying": "tie",
    "idly": "idl",
    "gently": "gentl",
    "ugly": "ugli",
    "early": "earli",
    "only": "onli",
    "singly": "singl",
    "sky": "sky",
    "news": "news",
    "howe": "howe",
    "atlas": "atlas",
    "cosmos": "cosmos",
    "bias": "bias",
    "andes": "andes",
    "inning": "inning",
    "innings": "inning",
    "outing": "outing",
    "outings": "outing",
    "canning": "canning",
    "cannings": "canning",
    "herring": "herring",
    "herrings": "herring",
    "earring": "earring",
    "earrings": "earring",
    "proceed": "proceed",
    "proceeds": "proceed",
    "proceeded": "proceed",
    "proceeding": "proceed",
    "exceed": "exceed",
    "exceeds": "exceed",
    "exceeded": "exceed",
    "exceeding": "exceed",
    "succeed": "success",
    "succeeds": "success",
    "succeeded": "success",
    "succeeding": "success",
}

def __r1r2(word, vowels):
    r1 = ""
    r2 = ""
    for i in range(1, len(word)):
        if word[i] not in vowels and word[i - 1] in vowels:
            r1 = word[i + 1 :]
            break
    for i in range(1, len(r1)):
        if r1[i] not in vowels and r1[i - 1] in vowels:
            r2 = r1[i + 1 :]
            break
    return (r1, r2)
    
def __suffix_replace(original, old, new):
    return original[: -len(old)] + new

def __stem(word):
    if word in __special_words:
        return __special_words[word]
    elif len(word) <= 3:
        return word

    if word.startswith("y"):
        word = "".join(("Y", word[1:]))
    for i in range(1, len(word)):
        if word[i - 1] in __vowels and word[i] == "y":
            word = "".join((word[:i], "Y", word[i + 1 :]))

    step1a_vowel_found = False
    step1b_vowel_found = False

    r1 = ""
    r2 = ""

    if word.startswith(("gener", "commun", "arsen")):
        if word.startswith(("gener", "arsen")):
            r1 = word[5:]
        else:
            r1 = word[6:]
        for i in range(1, len(r1)):
            if r1[i] not in __vowels and r1[i - 1] in __vowels:
                r2 = r1[i + 1 :]
                break
    else:
        r1, r2 = __r1r2(word, __vowels)

    # STEP 1a
    for suffix in __step1a_suffixes:
        if word.endswith(suffix):
            if suffix == "sses":
                word = word[:-2]
                r1 = r1[:-2]
                r2 = r2[:-2]
            elif suffix in ("ied", "ies"):
                if len(word[: -len(suffix)]) > 1:
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                else:
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]
            elif suffix == "s":
                for letter in word[:-2]:
                    if letter in __vowels:
                        step1a_vowel_found = True
                        break
                if step1a_vowel_found:
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]
            break

    # STEP 1b
    for suffix in __step1b_suffixes:
        if word.endswith(suffix):
            if suffix in ("eed", "eedly"):
                if r1.endswith(suffix):
                    word = __suffix_replace(word, suffix, "ee")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ee")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ee")
                    else:
                        r2 = ""
            else:
                for letter in word[: -len(suffix)]:
                    if letter in __vowels:
                        step1b_vowel_found = True
                        break
                if step1b_vowel_found:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    if word.endswith(("at", "bl", "iz")):
                        word = "".join((word, "e"))
                        r1 = "".join((r1, "e"))
                        if len(word) > 5 or len(r1) >= 3:
                            r2 = "".join((r2, "e"))
                    elif word.endswith(__double_consonants):
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                    elif (
                        r1 == ""
                        and len(word) >= 3
                        and word[-1] not in __vowels
                        and word[-1] not in "wxY"
                        and word[-2] in __vowels
                        and word[-3] not in __vowels
                    ) or (
                        r1 == ""
                        and len(word) == 2
                        and word[0] in __vowels
                        and word[1] not in __vowels
                    ):
                        word = "".join((word, "e"))
                        if len(r1) > 0:
                            r1 = "".join((r1, "e"))
                        if len(r2) > 0:
                            r2 = "".join((r2, "e"))
            break

    # STEP 1c
    if len(word) > 2 and word[-1] in "yY" and word[-2] not in __vowels:
        word = "".join((word[:-1], "i"))
        if len(r1) >= 1:
            r1 = "".join((r1[:-1], "i"))
        else:
            r1 = ""
        if len(r2) >= 1:
            r2 = "".join((r2[:-1], "i"))
        else:
            r2 = ""

    # STEP 2
    for suffix in __step2_suffixes:
        if word.endswith(suffix):
            if r1.endswith(suffix):
                if (
                    suffix in ("entli", "fulli", "lessli", "tional")
                    or (suffix == "li" and word[-3] in __li_ending)
                ):
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                elif suffix in ("enci", "anci", "abli"):
                    word = "".join((word[:-1], "e"))
                    if len(r1) >= 1:
                        r1 = "".join((r1[:-1], "e"))
                    else:
                        r1 = ""
                    if len(r2) >= 1:
                        r2 = "".join((r2[:-1], "e"))
                    else:
                        r2 = ""
                elif suffix in ("izer", "ization"):
                    word = __suffix_replace(word, suffix, "ize")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ize")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ize")
                    else:
                        r2 = ""
                elif suffix in ("ational", "ation", "ator"):
                    word = __suffix_replace(word, suffix, "ate")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ate")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ate")
                    else:
                        r2 = "e"
                elif suffix in ("alism", "aliti", "alli"):
                    word = __suffix_replace(word, suffix, "al")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "al")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "al")
                    else:
                        r2 = ""
                elif suffix == "fulness":
                    word = word[:-4]
                    r1 = r1[:-4]
                    r2 = r2[:-4]
                elif suffix in ("ousli", "ousness"):
                    word = __suffix_replace(word, suffix, "ous")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ous")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ous")
                    else:
                        r2 = ""
                elif suffix in ("iveness", "iviti"):
                    word = __suffix_replace(word, suffix, "ive")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ive")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ive")
                    else:
                        r2 = "e"
                elif suffix in ("biliti", "bli"):
                    word = __suffix_replace(word, suffix, "ble")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ble")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ble")
                    else:
                        r2 = ""
                elif suffix == "ogi" and word[-4] == "l":
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]
            break

    # STEP 3
    for suffix in __step3_suffixes:
        if word.endswith(suffix):
            if r1.endswith(suffix):
                if suffix == "tional":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                elif suffix == "ational":
                    word = __suffix_replace(word, suffix, "ate")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ate")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ate")
                    else:
                        r2 = ""
                elif suffix == "alize":
                    word = word[:-3]
                    r1 = r1[:-3]
                    r2 = r2[:-3]
                elif suffix in ("icate", "iciti", "ical"):
                    word = __suffix_replace(word, suffix, "ic")
                    if len(r1) >= len(suffix):
                        r1 = __suffix_replace(r1, suffix, "ic")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = __suffix_replace(r2, suffix, "ic")
                    else:
                        r2 = ""
                elif suffix in ("ful", "ness"):
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                elif suffix == "ative" and r2.endswith(suffix):
                    word = word[:-5]
                    r1 = r1[:-5]
                    r2 = r2[:-5]
            break

    # STEP 4
    for suffix in __step4_suffixes:
        if word.endswith(suffix):
            if r2.endswith(suffix):
                if suffix == "ion":
                    if word[-4] in "st":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
            break

    # STEP 5
    if (
        (r2.endswith("l") and word[-2] == "l")
        or r2.endswith("e")
        or (
            r1.endswith("e")
            and len(word) >= 4
            and (
                word[-2] in __vowels
                or word[-2] in "wxY"
                or word[-3] not in __vowels
                or word[-4] in __vowels
            )
        )
    ):
        word = word[:-1]

    word = word.replace("Y", "y")

    # STEP 6
    for suffix in __step6_suffixes:
        if word.endswith(suffix):
            if (
                (suffix == "graphi" and len(word) >= 9)
                or (suffix == "logi" and len(word) >= 7)
                or (suffix == "nomi" and len(word) >= 7)
                or (suffix == "pathi" and len(word) >= 6)
                or (suffix == "scopi" and len(word) >= 8)
                or (suffix == "therapi")
                or (suffix == "tomi" and len(word) >= 7)
                or (suffix == "tri" and len(word) >= 8 and word[-4] in "ae")
            ):
                word = word[:-1]
            elif suffix == "pathet" and len(word) >= 7:
                word = word[:-2]
            elif (
                (suffix == "curist" and len(word) >= 8)
                or (suffix == "logist" and len(word) >= 9)
                or (suffix == "nomist" and len(word) >= 9)
                or (suffix == "therapeut")
                or (suffix == "therapist")
                or (suffix == "tomist" and len(word) >= 9)
                or (suffix == "trist" and len(word) >= 10 and word[-6] in "ae")
                or (suffix == "turist" and len(word) >= 8)
            ):
                word = word[:-3]
            elif (
                (suffix == "physicist")
                or (suffix == "trician" and len(word) >= 10 and word[-8] in "ae")
            ):
                word = word[:-5]
            break

    return word

##########################################################
###   Class imbalance degree                           ###
###                                                    ###
###   Ortigosa-Hernandez, Inza, and Lozano (2017)      ###
###   https://github.com/jonathanSS/ImbalanceDegree/   ###
###   imbalancedegree.py                               ###
##########################################################

# Manhattan (or taxi-cab) distance 
def MANHATTAN_DISTANCE(multinomialDistribution):
	value = 0.0
	for parameter in multinomialDistribution:
		value += math.fabs(parameter - 1.0/len(multinomialDistribution))
	return value

# Euclidean distance
def EUCLIDEAN_DISTANCE(multinomialDistribution):
	value = 0.0
	for parameter in multinomialDistribution:
		value += (parameter - 1.0/len(multinomialDistribution))**2
	return math.sqrt(value)

# Chebyshev distance
def CHEBYSHEV_DISTANCE(multinomialDistribution):
	values = []
	for parameter in multinomialDistribution:
		values.append(math.fabs(parameter - 1.0/len(multinomialDistribution)))
	return max(values)

# Kullback-Leibler divergence
def KULLBACKLEIBLER_DIVERGENCE(multinomialDistribution):
	value = 0.0
	for parameter in multinomialDistribution:
		if parameter == 0.0:
			value += 0.0
		else:
			value += parameter*math.log(len(multinomialDistribution)*parameter,2)
	return value

# Helliger or Bhattacharyya distance
def HELLIGER_DISTANCE(multinomialDistribution):
	value = 0.0
	for parameter in multinomialDistribution:
		value += (math.sqrt(parameter) - math.sqrt(1.0/len(multinomialDistribution)))**2
	return 1./math.sqrt(2)*math.sqrt(value)

# Total Variation distance
def TOTALVARIATION_DISTANCE(multinomialDistribution):
	value = 0.0
	for parameter in multinomialDistribution:
		value += math.fabs(parameter - 1.0/len(multinomialDistribution))
	return 0.5*value

# Chi-square divergence
def CHISQUARE_DIVERGENCE(multinomialDistribution):
	value = 0.0
	for parameter in multinomialDistribution:
		value += (parameter - 1.0/len(multinomialDistribution))**2
	return len(multinomialDistribution)*value

class ImbalanceDegree(object):

	def __init__(self, classDistribution, complementMeasure="EUCLIDEAN_DISTANCE"):

		# Normalise the class distribution in the event of introducing the occurrences of the classes instead of the class distribution
		def normalisedDistribution(classDistribution):
			if sum(classDistribution) != 1.0:
				classDistribution = [1.0 * i / sum(classDistribution) for i in classDistribution]
			return classDistribution

		def balancedClassDistribution(k):
			return [1./k for i in range(k)]

		def distributionlowestEntropy(m, k):
			if m == 0:
				return balancedClassDistribution(k)
			else:
				distribution = [1.0 - (k - m - 1)*1./k]
				distribution.extend([0.0 for i in range(m)])
				distribution.extend([1./k for i in range(k - m - 1)])
				return distribution

		def applyComplementMeasure(complementMeasure, args):
			return eval(complementMeasure)(args[:])

		def imbalanceDegreeCalculator(numMinorityClasses, complementMeasure, classDistribution, balancedClassDistribution, lowestEntropyClassDistribution):
			if set(lowestEntropyClassDistribution) == set(balancedClassDistribution):
				return 0.0
			else:
				return (numMinorityClasses - 1 + applyComplementMeasure(complementMeasure, classDistribution) / applyComplementMeasure(complementMeasure, lowestEntropyClassDistribution))

		self.numClasses = len(classDistribution)
		self.numMinorityClasses = sum(i < (1.0/self.numClasses) for i in normalisedDistribution(classDistribution))
		self.complementMeasure = complementMeasure
		self.classDistribution = normalisedDistribution(classDistribution)
		self.balancedClassDistribution = [1.0/self.numClasses for i in range(self.numClasses)]
		self.lowestEntropyClassDistribution = distributionlowestEntropy(self.numMinorityClasses, self.numClasses)
		self.value = imbalanceDegreeCalculator(self.numMinorityClasses, self.complementMeasure, self.classDistribution, self.balancedClassDistribution, self.lowestEntropyClassDistribution)

#####################
###   Functions   ###
#####################

def print_banner(text):
    n = len(text)
    print("#" * (n + 12))
    print("###   " + text + "   ###")
    print("#" * (n + 12))
    return

def preprocess_text(text):
    t = text
    t = t.upper()
    t = t.strip()
    return t

def clean_text(text):
    t = text
    t = t.lower()
    t = re.sub(r"[^a-z]", r" ", t)
    t = re.sub(r"\s+", r" ", t)
    t = t.strip()
    t = " ".join(word for word in t.split(" ") if word not in __stop_words)
    t = " ".join(__stem(word) for word in t.split(" ") if word != "")
    return t

def get_words(text):
    if text == "":
        return []
    return list(set(text.split(" ")))

def get_grams2(text):
    grams2 = set()
    words  = get_words(text)
    n      = len(words)
    if n >= 2:
        for i in range(n - 1):
            gram2 = "GRAM2|" + "_".join([words[i], words[i + 1]])
            grams2.add(gram2)
    return list(grams2)

def get_grams3(text):
    grams3 = set()
    words  = get_words(text)
    n      = len(words)
    if n >= 3:
        for i in range(n - 2):
            gram3 = "GRAM3|" + "_".join([words[i], words[i + 1], words[i + 2]])
            grams3.add(gram3)
    return list(grams3)

def get_combs2(text):
    combs2 = set()
    words  = get_words(text)
    n      = len(words)
    if n >= 2:
        for i in range(n - 1):
            for j in range(i + 1, n):
                comb2 = "COMB2|" + "_".join(sorted([words[i], words[j]]))
                combs2.add(comb2)
    return list(combs2)

def get_combs3(text):
    combs3 = set()
    words  = get_words(text)
    n      = len(words)
    if n >= 3:
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    comb3 = "COMB3|" + "_".join(sorted([words[i], words[j], words[k]]))
                    combs3.add(comb3)
    return list(combs3)

def calc_wordbit_freqs_detail(wordbits_train, y_train, n_obs_train, unique_y_full):
    wordbit_freqs_detail = {}
    for i in range(n_obs_train):
        wordbits = wordbits_train[i]
        y        = y_train[i]
        for wordbit in wordbits:
            if wordbit not in wordbit_freqs_detail:
                wordbit_freqs_detail[wordbit] = {y_temp: 0 for y_temp in unique_y_full}
            wordbit_freqs_detail[wordbit][y] += 1
    return wordbit_freqs_detail

def calc_wordbit_freqs(wordbit_freqs_detail):
    wordbit_freqs = {}
    for wordbit in wordbit_freqs_detail:
        wordbit_freqs[wordbit] = 0
        for y in wordbit_freqs_detail[wordbit]:
            wordbit_freqs[wordbit] += wordbit_freqs_detail[wordbit][y]
    return wordbit_freqs

def calc_wordbit_doms(wordbit_freqs_detail, wordbit_freqs, unique_y_full):
    n_y         = 1.0 * len(unique_y_full)
    wordbit_doms = {}
    for wordbit in wordbit_freqs_detail:
        props                = [wordbit_freqs_detail[wordbit][y] / wordbit_freqs[wordbit] for y in unique_y_full]
        wordbit_doms[wordbit] = max(props)
    return wordbit_doms

def get_feats(wordbits, wordbits_top):
    feats = {}
    for wordbit in wordbits:
        if wordbit in wordbits_top:
            feats[wordbit] = True
    return feats

def calc_acc(truths, preds):
    return accuracy_score(truths, preds)

def calc_f1(truths, preds):
    return f1_score(truths, preds, average="weighted")

def main():

    ##########################
    ###   Set parameters   ###
    ##########################

    print("")
    print_banner("Set parameters")

    dataset         = "data"
    train_frac      = 0.8
    top_freq_thresh = 1
    top_dom_thresh  = 0.4

    print("")
    print("{:<30}{}".format("Dataset:",                  dataset))
    print("{:<30}{}".format("Training fraction:",        train_frac))
    print("{:<30}{}".format("Feature frequency thresh:", top_freq_thresh))
    print("{:<30}{}".format("Feature dominance thresh:", top_dom_thresh))

    ###################################################
    ###   Read in dataset                           ###
    ###   -- HuffPost article headlines             ###
    ###   -- NAICS write-in business descriptions   ###
    ###   -- Stack Overflow thread titles           ###
    ###################################################

    print("")
    print_banner("Read in dataset")

    id_full       = []
    text_raw_full = []
    y_full        = []

    count = 0
    f = io.open("{}.txt".format(dataset), "r")
    for line in f:
        count += 1
        if count >= 2:
            line_match = line.strip().split("|")
            if len(line_match) == 2:
                id       = str(count - 1).zfill(6)
                text_raw = preprocess_text(line_match[0])
                y        = line_match[1]
                id_full.append(id)
                text_raw_full.append(text_raw)
                y_full.append(y)
    f.close()

    del count, f, id, line, line_match, text_raw, y

    #########################################
    ###   Apply text cleaning algorithm   ###
    ###   -- Remove non-letters           ###
    ###   -- Remove extra whitespace      ###
    ###   -- Remove stop words            ###
    ###   -- Stem words                   ###
    #########################################

    print("")
    print_banner("Apply text cleaning algorithm")

    text_clean_full = [clean_text(text) for text in text_raw_full]

    #############################################
    ###   Summarize dataset                   ###
    ###   -- Number of observations           ###
    ###   -- Number of classes                ###
    ###   -- Class imbalance degree           ###
    ###   -- Class imbalance ratio            ###
    ###   -- Raw text length (characters)     ###
    ###   -- Raw text length (words)          ###
    ###   -- Clean text length (characters)   ###
    ###   -- Clean text length (words)        ###
    #############################################

    print("")
    print_banner("Summarize dataset")

    n_obs         = len(y_full)
    unique_y_full = sorted(set(y_full))
    n_classes     = len(unique_y_full)

    class_freqs_dict = {}
    for y in y_full:
        if y not in class_freqs_dict:
            class_freqs_dict[y] = 0
        class_freqs_dict[y] += 1
    class_freqs = [class_freqs_dict[y] for y in class_freqs_dict]

    ci_object = ImbalanceDegree(class_freqs)
    ci_degree = round(ci_object.value / (n_classes - 1), 2)
    ci_ratio  = round(max(class_freqs) / min(class_freqs), 2)

    len_text_raw_char   = [len(text)            for text in text_raw_full]
    len_text_raw_word   = [len(text.split(" ")) for text in text_raw_full]
    len_text_clean_char = [len(text)            for text in text_clean_full]
    len_text_clean_word = [len(text.split(" ")) for text in text_clean_full]

    avg_len_text_raw_char   = round(statistics.mean(len_text_raw_char),   2)
    med_len_text_raw_char   = round(statistics.median(len_text_raw_char), 2)
    max_len_text_raw_char   = max(len_text_raw_char)

    avg_len_text_raw_word   = round(statistics.mean(len_text_raw_word),   2)
    med_len_text_raw_word   = round(statistics.median(len_text_raw_word), 2)
    max_len_text_raw_word   = max(len_text_raw_word)

    avg_len_text_clean_char = round(statistics.mean(len_text_clean_char),   2)
    med_len_text_clean_char = round(statistics.median(len_text_clean_char), 2)
    max_len_text_clean_char = max(len_text_clean_char)

    avg_len_text_clean_word = round(statistics.mean(len_text_clean_word),   2)
    med_len_text_clean_word = round(statistics.median(len_text_clean_word), 2)
    max_len_text_clean_word = max(len_text_clean_word)

    del class_freqs, len_text_clean_char, len_text_clean_word, len_text_raw_char, len_text_raw_word, y

    print("")
    print("{:<30}{}".format("Number of obs:",          n_obs))
    print("{:<30}{}".format("Number of classes:",      n_classes))
    print("{:<30}{}".format("Class imbalance degree:", ci_degree))
    print("{:<30}{}".format("Class imbalance ratio:",  ci_ratio))
    print("")
    print("{}".format("Raw text length (chars)"))
    print("{}{:<28}{}".format("  ", "Average:", avg_len_text_raw_char))
    print("{}{:<28}{}".format("  ", "Median:",  med_len_text_raw_char))
    print("{}{:<28}{}".format("  ", "Maximum:", max_len_text_raw_char))
    print("")
    print("{}".format("Raw text length (words)"))
    print("{}{:<28}{}".format("  ", "Average:", avg_len_text_raw_word))
    print("{}{:<28}{}".format("  ", "Median:",  med_len_text_raw_word))
    print("{}{:<28}{}".format("  ", "Maximum:", max_len_text_raw_word))
    print("")
    print("{}".format("Clean text length (chars)"))
    print("{}{:<28}{}".format("  ", "Average:", avg_len_text_clean_char))
    print("{}{:<28}{}".format("  ", "Median:",  med_len_text_clean_char))
    print("{}{:<28}{}".format("  ", "Maximum:", max_len_text_clean_char))
    print("")
    print("{}".format("Clean text length (words)"))
    print("{}{:<28}{}".format("  ", "Average:", avg_len_text_clean_word))
    print("{}{:<28}{}".format("  ", "Median:",  med_len_text_clean_word))
    print("{}{:<28}{}".format("  ", "Maximum:", max_len_text_clean_word))

    #####################################################
    ###   Split dataset into training and test sets   ###
    ###   -- Stratify by class                        ###
    ###   -- 80%-20%                                  ###
    #####################################################

    print("")
    print_banner("Split dataset into training and test sets")

    sample_freqs_dict = {}
    sample_sizes_dict = {}
    for y in class_freqs_dict:
        sample_freqs_dict[y] = 0
        sample_sizes_dict[y] = int(round(class_freqs_dict[y] * train_frac))

    data_full = [(z[0], z[1], z[2], z[3]) for z in zip(id_full, text_raw_full, text_clean_full, y_full)]
    random.seed(12345)
    random.shuffle(data_full)

    data_train = []
    data_test  = []

    for z in data_full:
        y = z[3]
        if sample_freqs_dict[y] < sample_sizes_dict[y]:
            data_train.append(z)
            sample_freqs_dict[y] += 1
        else:
            data_test.append(z)

    data_train.sort(key=lambda z: z[0])
    data_test.sort(key=lambda z: z[0])

    id_train         = []
    text_raw_train   = []
    text_clean_train = []
    y_train          = []

    for z in data_train:
        id_train.append(z[0])
        text_raw_train.append(z[1])
        text_clean_train.append(z[2])
        y_train.append(z[3])

    id_test         = []
    text_raw_test   = []
    text_clean_test = []
    y_test          = []

    for z in data_test:
        id_test.append(z[0])
        text_raw_test.append(z[1])
        text_clean_test.append(z[2])
        y_test.append(z[3])

    n_obs_train = len(y_train)
    n_obs_test  = len(y_test)

    del data_full, data_test, data_train, sample_freqs_dict, y, z

    print("")
    print("{:<30}{}".format("Number of training obs:", n_obs_train))
    print("{:<30}{}".format("Number of test obs:",     n_obs_test))

    #################################################
    ###   Determine features using training set   ###
    #################################################

    print("")
    print_banner("Determine features using training set")

    words_train  = [get_words(text)  for text in text_clean_train]
    grams2_train = [get_grams2(text) for text in text_clean_train]
    grams3_train = [get_grams3(text) for text in text_clean_train]
    combs2_train = [get_combs2(text) for text in text_clean_train]
    combs3_train = [get_combs3(text) for text in text_clean_train]

    words_freqs_detail  = calc_wordbit_freqs_detail(words_train,  y_train, n_obs_train, unique_y_full)
    grams2_freqs_detail = calc_wordbit_freqs_detail(grams2_train, y_train, n_obs_train, unique_y_full)
    grams3_freqs_detail = calc_wordbit_freqs_detail(grams3_train, y_train, n_obs_train, unique_y_full)
    combs2_freqs_detail = calc_wordbit_freqs_detail(combs2_train, y_train, n_obs_train, unique_y_full)
    combs3_freqs_detail = calc_wordbit_freqs_detail(combs3_train, y_train, n_obs_train, unique_y_full)

    words_freqs  = calc_wordbit_freqs(words_freqs_detail)
    grams2_freqs = calc_wordbit_freqs(grams2_freqs_detail)
    grams3_freqs = calc_wordbit_freqs(grams3_freqs_detail)
    combs2_freqs = calc_wordbit_freqs(combs2_freqs_detail)
    combs3_freqs = calc_wordbit_freqs(combs3_freqs_detail)

    words_doms  = calc_wordbit_doms(words_freqs_detail,  words_freqs,  unique_y_full)
    grams2_doms = calc_wordbit_doms(grams2_freqs_detail, grams2_freqs, unique_y_full)
    grams3_doms = calc_wordbit_doms(grams3_freqs_detail, grams3_freqs, unique_y_full)
    combs2_doms = calc_wordbit_doms(combs2_freqs_detail, combs2_freqs, unique_y_full)
    combs3_doms = calc_wordbit_doms(combs3_freqs_detail, combs3_freqs, unique_y_full)

    words_top  = sorted([word  for word  in words_freqs  if words_freqs[word]   >= top_freq_thresh and words_doms[word]   >= top_dom_thresh])
    grams2_top = sorted([gram2 for gram2 in grams2_freqs if grams2_freqs[gram2] >= top_freq_thresh and grams2_doms[gram2] >= top_dom_thresh])
    grams3_top = sorted([gram3 for gram3 in grams3_freqs if grams3_freqs[gram3] >= top_freq_thresh and grams3_doms[gram3] >= top_dom_thresh])
    combs2_top = sorted([comb2 for comb2 in combs2_freqs if combs2_freqs[comb2] >= top_freq_thresh and combs2_doms[comb2] >= top_dom_thresh])
    combs3_top = sorted([comb3 for comb3 in combs3_freqs if combs3_freqs[comb3] >= top_freq_thresh and combs3_doms[comb3] >= top_dom_thresh])

    n_wordbits     = len(words_freqs) + len(grams2_freqs) + len(grams3_freqs) + len(combs2_freqs) + len(combs3_freqs)
    n_wordbits_top = len(words_top)   + len(grams2_top)   + len(grams3_top)   + len(combs2_top)   + len(combs3_top)

    print("")
    print("{}".format("Words"))
    print("{}{:<28}{}".format("  ", "Number:",                  len(words_freqs)))
    print("{}{:<28}{}".format("  ", "Number used as features:", len(words_top)))
    print("{}{:<28}{}".format("  ", "Propor used as features:", round(len(words_top) / len(words_freqs), 4)))
    print("")
    print("{}".format("2-grams"))
    print("{}{:<28}{}".format("  ", "Number:",                  len(grams2_freqs)))
    print("{}{:<28}{}".format("  ", "Number used as features:", len(grams2_top)))
    print("{}{:<28}{}".format("  ", "Propor used as features:", round(len(grams2_top) / len(grams2_freqs), 4)))
    print("")
    print("{}".format("3-grams"))
    print("{}{:<28}{}".format("  ", "Number:",                  len(grams3_freqs)))
    print("{}{:<28}{}".format("  ", "Number used as features:", len(grams3_top)))
    print("{}{:<28}{}".format("  ", "Propor used as features:", round(len(grams3_top) / len(grams3_freqs), 4)))
    print("")
    print("{}".format("2-combs"))
    print("{}{:<28}{}".format("  ", "Number:",                  len(combs2_freqs)))
    print("{}{:<28}{}".format("  ", "Number used as features:", len(combs2_top)))
    print("{}{:<28}{}".format("  ", "Propor used as features:", round(len(combs2_top) / len(combs2_freqs), 4)))
    print("")
    print("{}".format("3-combs"))
    print("{}{:<28}{}".format("  ", "Number:",                  len(combs3_freqs)))
    print("{}{:<28}{}".format("  ", "Number used as features:", len(combs3_top)))
    print("{}{:<28}{}".format("  ", "Propor used as features:", round(len(combs3_top) / len(combs3_freqs), 4)))
    print("")
    print("{}".format("Total"))
    print("{}{:<28}{}".format("  ", "Number:",                  n_wordbits))
    print("{}{:<28}{}".format("  ", "Number used as features:", n_wordbits_top))
    print("{}{:<28}{}".format("  ", "Propor used as features:", round(n_wordbits_top / n_wordbits, 4)))

    ############################################
    ###   Create X values for training set   ###
    ############################################

    print("")
    print_banner("Create X values for training set")

    X_w_train  = [get_feats(words,  words_top)  for words  in words_train]
    X_g2_train = [get_feats(grams2, grams2_top) for grams2 in grams2_train]
    X_g3_train = [get_feats(grams3, grams3_top) for grams3 in grams3_train]
    X_c2_train = [get_feats(combs2, combs2_top) for combs2 in combs2_train]
    X_c3_train = [get_feats(combs3, combs3_top) for combs3 in combs3_train]

    X_w_g2_train = []
    for i in range(n_obs_train):
        d = {}
        d.update(X_w_train[i])
        d.update(X_g2_train[i])
        X_w_g2_train.append(d)

    X_w_g23_train = []
    for i in range(n_obs_train):
        d = {}
        d.update(X_w_train[i])
        d.update(X_g2_train[i])
        d.update(X_g3_train[i])
        X_w_g23_train.append(d)

    X_w_c2_train = []
    for i in range(n_obs_train):
        d = {}
        d.update(X_w_train[i])
        d.update(X_c2_train[i])
        X_w_c2_train.append(d)

    X_w_c23_train = []
    for i in range(n_obs_train):
        d = {}
        d.update(X_w_train[i])
        d.update(X_c2_train[i])
        d.update(X_c3_train[i])
        X_w_c23_train.append(d)

    X_w_g2_c2_train = []
    for i in range(n_obs_train):
        d = {}
        d.update(X_w_train[i])
        d.update(X_g2_train[i])
        d.update(X_c2_train[i])
        X_w_g2_c2_train.append(d)

    X_w_g23_c23_train = []
    for i in range(n_obs_train):
        d = {}
        d.update(X_w_train[i])
        d.update(X_g2_train[i])
        d.update(X_g3_train[i])
        d.update(X_c2_train[i])
        d.update(X_c3_train[i])
        X_w_g23_c23_train.append(d)

    Xy_w_train         = [(X_w_train[i],         y_train[i]) for i in range(n_obs_train)]
    Xy_w_g2_train      = [(X_w_g2_train[i],      y_train[i]) for i in range(n_obs_train)]
    Xy_w_g23_train     = [(X_w_g23_train[i],     y_train[i]) for i in range(n_obs_train)]
    Xy_w_c2_train      = [(X_w_c2_train[i],      y_train[i]) for i in range(n_obs_train)]
    Xy_w_c23_train     = [(X_w_c23_train[i],     y_train[i]) for i in range(n_obs_train)]
    Xy_w_g2_c2_train   = [(X_w_g2_c2_train[i],   y_train[i]) for i in range(n_obs_train)]
    Xy_w_g23_c23_train = [(X_w_g23_c23_train[i], y_train[i]) for i in range(n_obs_train)]

    del d, i

    #########################################
    ###   Fit models using training set   ###
    #########################################

    print("")
    print_banner("Fit models using training set")

    # Decision tree
    mod_DT_w = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w.train(Xy_w_train)
    mod_DT_w_g2 = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w_g2.train(Xy_w_g2_train)
    mod_DT_w_g23 = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w_g23.train(Xy_w_g23_train)
    mod_DT_w_c2 = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w_c2.train(Xy_w_c2_train)
    mod_DT_w_c23 = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w_c23.train(Xy_w_c23_train)
    mod_DT_w_g2_c2 = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w_g2_c2.train(Xy_w_g2_c2_train)
    mod_DT_w_g23_c23 = nltk.classify.SklearnClassifier(DecisionTreeClassifier())
    mod_DT_w_g23_c23.train(Xy_w_g23_c23_train)

    # Logistic regression
    mod_LR_w = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w.train(Xy_w_train)
    mod_LR_w_g2 = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w_g2.train(Xy_w_g2_train)
    mod_LR_w_g23 = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w_g23.train(Xy_w_g23_train)
    mod_LR_w_c2 = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w_c2.train(Xy_w_c2_train)
    mod_LR_w_c23 = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w_c23.train(Xy_w_c23_train)
    mod_LR_w_g2_c2 = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w_g2_c2.train(Xy_w_g2_c2_train)
    mod_LR_w_g23_c23 = nltk.classify.SklearnClassifier(LogisticRegression())
    mod_LR_w_g23_c23.train(Xy_w_g23_c23_train)

    # Neural network
    mod_NN_w = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w.train(Xy_w_train)
    mod_NN_w_g2 = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w_g2.train(Xy_w_g2_train)
    mod_NN_w_g23 = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w_g23.train(Xy_w_g23_train)
    mod_NN_w_c2 = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w_c2.train(Xy_w_c2_train)
    mod_NN_w_c23 = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w_c23.train(Xy_w_c23_train)
    mod_NN_w_g2_c2 = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w_g2_c2.train(Xy_w_g2_c2_train)
    mod_NN_w_g23_c23 = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=500))
    mod_NN_w_g23_c23.train(Xy_w_g23_c23_train)

    print("")
    print("Decision tree")
    print("Logistic regression")
    print("Neural network")

    ########################################
    ###   Create X values for test set   ###
    ########################################

    print("")
    print_banner("Create X values for test set")

    words_test  = [get_words(text)  for text in text_clean_test]
    grams2_test = [get_grams2(text) for text in text_clean_test]
    grams3_test = [get_grams3(text) for text in text_clean_test]
    combs2_test = [get_combs2(text) for text in text_clean_test]
    combs3_test = [get_combs3(text) for text in text_clean_test]

    X_w_test  = [get_feats(words,  words_top)  for words  in words_test]
    X_g2_test = [get_feats(grams2, grams2_top) for grams2 in grams2_test]
    X_g3_test = [get_feats(grams3, grams3_top) for grams3 in grams3_test]
    X_c2_test = [get_feats(combs2, combs2_top) for combs2 in combs2_test]
    X_c3_test = [get_feats(combs3, combs3_top) for combs3 in combs3_test]

    X_w_g2_test = []
    for i in range(n_obs_test):
        d = {}
        d.update(X_w_test[i])
        d.update(X_g2_test[i])
        X_w_g2_test.append(d)

    X_w_g23_test = []
    for i in range(n_obs_test):
        d = {}
        d.update(X_w_test[i])
        d.update(X_g2_test[i])
        d.update(X_g3_test[i])
        X_w_g23_test.append(d)

    X_w_c2_test = []
    for i in range(n_obs_test):
        d = {}
        d.update(X_w_test[i])
        d.update(X_c2_test[i])
        X_w_c2_test.append(d)

    X_w_c23_test = []
    for i in range(n_obs_test):
        d = {}
        d.update(X_w_test[i])
        d.update(X_c2_test[i])
        d.update(X_c3_test[i])
        X_w_c23_test.append(d)

    X_w_g2_c2_test = []
    for i in range(n_obs_test):
        d = {}
        d.update(X_w_test[i])
        d.update(X_g2_test[i])
        d.update(X_c2_test[i])
        X_w_g2_c2_test.append(d)

    X_w_g23_c23_test = []
    for i in range(n_obs_test):
        d = {}
        d.update(X_w_test[i])
        d.update(X_g2_test[i])
        d.update(X_g3_test[i])
        d.update(X_c2_test[i])
        d.update(X_c3_test[i])
        X_w_g23_c23_test.append(d)

    del d, i

    ###########################################
    ###   Apply fitted models to test set   ###
    ###########################################

    print("")
    print_banner("Apply fitted models to test set")

    # Decision tree
    preds_DT_w         = [mod_DT_w.classify(X)         for X in X_w_test]
    preds_DT_w_g2      = [mod_DT_w_g2.classify(X)      for X in X_w_g2_test]
    preds_DT_w_g23     = [mod_DT_w_g23.classify(X)     for X in X_w_g23_test]
    preds_DT_w_c2      = [mod_DT_w_c2.classify(X)      for X in X_w_c2_test]
    preds_DT_w_c23     = [mod_DT_w_c23.classify(X)     for X in X_w_c23_test]
    preds_DT_w_g2_c2   = [mod_DT_w_g2_c2.classify(X)   for X in X_w_g2_c2_test]
    preds_DT_w_g23_c23 = [mod_DT_w_g23_c23.classify(X) for X in X_w_g23_c23_test]

    # Logistic regression
    preds_LR_w         = [mod_LR_w.classify(X)         for X in X_w_test]
    preds_LR_w_g2      = [mod_LR_w_g2.classify(X)      for X in X_w_g2_test]
    preds_LR_w_g23     = [mod_LR_w_g23.classify(X)     for X in X_w_g23_test]
    preds_LR_w_c2      = [mod_LR_w_c2.classify(X)      for X in X_w_c2_test]
    preds_LR_w_c23     = [mod_LR_w_c23.classify(X)     for X in X_w_c23_test]
    preds_LR_w_g2_c2   = [mod_LR_w_g2_c2.classify(X)   for X in X_w_g2_c2_test]
    preds_LR_w_g23_c23 = [mod_LR_w_g23_c23.classify(X) for X in X_w_g23_c23_test]

    # Neural network
    preds_NN_w         = [mod_NN_w.classify(X)         for X in X_w_test]
    preds_NN_w_g2      = [mod_NN_w_g2.classify(X)      for X in X_w_g2_test]
    preds_NN_w_g23     = [mod_NN_w_g23.classify(X)     for X in X_w_g23_test]
    preds_NN_w_c2      = [mod_NN_w_c2.classify(X)      for X in X_w_c2_test]
    preds_NN_w_c23     = [mod_NN_w_c23.classify(X)     for X in X_w_c23_test]
    preds_NN_w_g2_c2   = [mod_NN_w_g2_c2.classify(X)   for X in X_w_g2_c2_test]
    preds_NN_w_g23_c23 = [mod_NN_w_g23_c23.classify(X) for X in X_w_g23_c23_test]

    ###############################################
    ###   Calculate model performance metrics   ###
    ###   -- Accuracy                           ###
    ###   -- F1-score                           ###
    ###############################################

    print("")
    print_banner("Calculate accuracies")

     # Decision tree
    acc_DT_w         = calc_acc(y_test, preds_DT_w)
    acc_DT_w_g2      = calc_acc(y_test, preds_DT_w_g2)
    acc_DT_w_g23     = calc_acc(y_test, preds_DT_w_g23)
    acc_DT_w_c2      = calc_acc(y_test, preds_DT_w_c2)
    acc_DT_w_c23     = calc_acc(y_test, preds_DT_w_c23)
    acc_DT_w_g2_c2   = calc_acc(y_test, preds_DT_w_g2_c2)
    acc_DT_w_g23_c23 = calc_acc(y_test, preds_DT_w_g23_c23)

    # Logistic regression
    acc_LR_w         = calc_acc(y_test, preds_LR_w)
    acc_LR_w_g2      = calc_acc(y_test, preds_LR_w_g2)
    acc_LR_w_g23     = calc_acc(y_test, preds_LR_w_g23)
    acc_LR_w_c2      = calc_acc(y_test, preds_LR_w_c2)
    acc_LR_w_c23     = calc_acc(y_test, preds_LR_w_c23)
    acc_LR_w_g2_c2   = calc_acc(y_test, preds_LR_w_g2_c2)
    acc_LR_w_g23_c23 = calc_acc(y_test, preds_LR_w_g23_c23)

    # Neural network
    acc_NN_w         = calc_acc(y_test, preds_NN_w)
    acc_NN_w_g2      = calc_acc(y_test, preds_NN_w_g2)
    acc_NN_w_g23     = calc_acc(y_test, preds_NN_w_g23)
    acc_NN_w_c2      = calc_acc(y_test, preds_NN_w_c2)
    acc_NN_w_c23     = calc_acc(y_test, preds_NN_w_c23)
    acc_NN_w_g2_c2   = calc_acc(y_test, preds_NN_w_g2_c2)
    acc_NN_w_g23_c23 = calc_acc(y_test, preds_NN_w_g23_c23)

    print("")
    print("{}".format("Decision tree"))
    print("{}{:<15}{}".format("  ", "w",         round(acc_DT_w,         4)))
    print("{}{:<15}{}".format("  ", "w g2",      round(acc_DT_w_g2,      4)))
    print("{}{:<15}{}".format("  ", "w g23",     round(acc_DT_w_g23,     4)))
    print("{}{:<15}{}".format("  ", "w     c2",  round(acc_DT_w_c2,      4)))
    print("{}{:<15}{}".format("  ", "w     c23", round(acc_DT_w_c23,     4)))
    print("{}{:<15}{}".format("  ", "w g2  c2",  round(acc_DT_w_g2_c2,   4)))
    print("{}{:<15}{}".format("  ", "w g23 c23", round(acc_DT_w_g23_c23, 4)))
    print("")
    print("{}".format("Logistic regression"))
    print("{}{:<15}{}".format("  ", "w",         round(acc_LR_w,         4)))
    print("{}{:<15}{}".format("  ", "w g2",      round(acc_LR_w_g2,      4)))
    print("{}{:<15}{}".format("  ", "w g23",     round(acc_LR_w_g23,     4)))
    print("{}{:<15}{}".format("  ", "w     c2",  round(acc_LR_w_c2,      4)))
    print("{}{:<15}{}".format("  ", "w     c23", round(acc_LR_w_c23,     4)))
    print("{}{:<15}{}".format("  ", "w g2  c2",  round(acc_LR_w_g2_c2,   4)))
    print("{}{:<15}{}".format("  ", "w g23 c23", round(acc_LR_w_g23_c23, 4)))
    print("")
    print("{}".format("Neural network"))
    print("{}{:<15}{}".format("  ", "w",         round(acc_NN_w,         4)))
    print("{}{:<15}{}".format("  ", "w g2",      round(acc_NN_w_g2,      4)))
    print("{}{:<15}{}".format("  ", "w g23",     round(acc_NN_w_g23,     4)))
    print("{}{:<15}{}".format("  ", "w     c2",  round(acc_NN_w_c2,      4)))
    print("{}{:<15}{}".format("  ", "w     c23", round(acc_NN_w_c23,     4)))
    print("{}{:<15}{}".format("  ", "w g2  c2",  round(acc_NN_w_g2_c2,   4)))
    print("{}{:<15}{}".format("  ", "w g23 c23", round(acc_NN_w_g23_c23, 4)))

    print("")
    print_banner("Calculate F1-scores")

    # Decision tree
    f1_DT_w         = calc_f1(y_test, preds_DT_w)
    f1_DT_w_g2      = calc_f1(y_test, preds_DT_w_g2)
    f1_DT_w_g23     = calc_f1(y_test, preds_DT_w_g23)
    f1_DT_w_c2      = calc_f1(y_test, preds_DT_w_c2)
    f1_DT_w_c23     = calc_f1(y_test, preds_DT_w_c23)
    f1_DT_w_g2_c2   = calc_f1(y_test, preds_DT_w_g2_c2)
    f1_DT_w_g23_c23 = calc_f1(y_test, preds_DT_w_g23_c23)

    # Logistic regression
    f1_LR_w         = calc_f1(y_test, preds_LR_w)
    f1_LR_w_g2      = calc_f1(y_test, preds_LR_w_g2)
    f1_LR_w_g23     = calc_f1(y_test, preds_LR_w_g23)
    f1_LR_w_c2      = calc_f1(y_test, preds_LR_w_c2)
    f1_LR_w_c23     = calc_f1(y_test, preds_LR_w_c23)
    f1_LR_w_g2_c2   = calc_f1(y_test, preds_LR_w_g2_c2)
    f1_LR_w_g23_c23 = calc_f1(y_test, preds_LR_w_g23_c23)

    # Neural network
    f1_NN_w         = calc_f1(y_test, preds_NN_w)
    f1_NN_w_g2      = calc_f1(y_test, preds_NN_w_g2)
    f1_NN_w_g23     = calc_f1(y_test, preds_NN_w_g23)
    f1_NN_w_c2      = calc_f1(y_test, preds_NN_w_c2)
    f1_NN_w_c23     = calc_f1(y_test, preds_NN_w_c23)
    f1_NN_w_g2_c2   = calc_f1(y_test, preds_NN_w_g2_c2)
    f1_NN_w_g23_c23 = calc_f1(y_test, preds_NN_w_g23_c23)

    print("")
    print("{}".format("Decision tree"))
    print("{}{:<15}{}".format("  ", "w",         round(f1_DT_w,         4)))
    print("{}{:<15}{}".format("  ", "w g2",      round(f1_DT_w_g2,      4)))
    print("{}{:<15}{}".format("  ", "w g23",     round(f1_DT_w_g23,     4)))
    print("{}{:<15}{}".format("  ", "w     c2",  round(f1_DT_w_c2,      4)))
    print("{}{:<15}{}".format("  ", "w     c23", round(f1_DT_w_c23,     4)))
    print("{}{:<15}{}".format("  ", "w g2  c2",  round(f1_DT_w_g2_c2,   4)))
    print("{}{:<15}{}".format("  ", "w g23 c23", round(f1_DT_w_g23_c23, 4)))
    print("")
    print("{}".format("Logistic regression"))
    print("{}{:<15}{}".format("  ", "w",         round(f1_LR_w,         4)))
    print("{}{:<15}{}".format("  ", "w g2",      round(f1_LR_w_g2,      4)))
    print("{}{:<15}{}".format("  ", "w g23",     round(f1_LR_w_g23,     4)))
    print("{}{:<15}{}".format("  ", "w     c2",  round(f1_LR_w_c2,      4)))
    print("{}{:<15}{}".format("  ", "w     c23", round(f1_LR_w_c23,     4)))
    print("{}{:<15}{}".format("  ", "w g2  c2",  round(f1_LR_w_g2_c2,   4)))
    print("{}{:<15}{}".format("  ", "w g23 c23", round(f1_LR_w_g23_c23, 4)))
    print("")
    print("{}".format("Neural network"))
    print("{}{:<15}{}".format("  ", "w",         round(f1_NN_w,         4)))
    print("{}{:<15}{}".format("  ", "w g2",      round(f1_NN_w_g2,      4)))
    print("{}{:<15}{}".format("  ", "w g23",     round(f1_NN_w_g23,     4)))
    print("{}{:<15}{}".format("  ", "w     c2",  round(f1_NN_w_c2,      4)))
    print("{}{:<15}{}".format("  ", "w     c23", round(f1_NN_w_c23,     4)))
    print("{}{:<15}{}".format("  ", "w g2  c2",  round(f1_NN_w_g2_c2,   4)))
    print("{}{:<15}{}".format("  ", "w g23 c23", round(f1_NN_w_g23_c23, 4)))

    print("")
    return

if __name__ == "__main__":
    main()
