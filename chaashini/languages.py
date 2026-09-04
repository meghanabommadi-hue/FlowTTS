"""Registry of target languages: codes, names, scripts, function words, discovery hints.

Scripts are detected with Unicode block regexes (see lid.py). Where one script serves
several languages (Devanagari, Bengali, Arabic, Kannada) short function-word lists break
the tie. The lists are deliberately high-frequency grammatical words, not content words.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str
    scripts: tuple[str, ...]
    stopwords: frozenset[str] = field(default_factory=frozenset)
    query_names: tuple[str, ...] = ()       # how people refer to the language in searches
    regions: tuple[str, ...] = ()           # geography flavour for query generation
    asr_supported: bool = True              # the multilingual ASR in use can transcribe it
    query_hint: str = ""                    # extra instructions for the query generator


def _sw(s: str) -> frozenset[str]:
    return frozenset(w for w in s.split() if w)


LANGUAGES: dict[str, Language] = {}


def _add(l: Language) -> None:
    LANGUAGES[l.code] = l


_add(Language("hi", "Hindi", "हिन्दी", ("devanagari",),
    _sw("का की के में है हैं को से और नहीं था थी थे यह वह हम आप तुम मैं हूँ हूं पर भी कि जो तो अब क्या कैसे लिए गया गई गए हो होता होती होते रहा रही रहे किया किए करना कर सकते सकता चाहिए लेकिन क्योंकि अगर जब तब उनका उनकी उसका उसकी इसका इसकी बहुत कुछ सब ये वो उन इन इस उस कोई कर्ता वाला वाली वाले"),
    ("Hindi", "हिंदी", "hindi"), ("Delhi", "Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan", "Uttarakhand", "Jharkhand", "Haryana")))
_add(Language("mr", "Marathi", "मराठी", ("devanagari",),
    _sw("आहे आहेत आणि मी तू तो ती ते त्या त्याचा त्याची त्यांचा होता होती होते नाही नव्हता मध्ये ला ना ने च्या चा ची चे कडे पण किंवा म्हणून म्हणजे कारण जर तर काय कसे केला केली केले करतो करते करतात असे अशी असा हे ही हा आम्ही तुम्ही आपण खूप सर्व झाला झाली झाले माझा माझी माझे तुझा तुझी तुझे आपला आपली आपले"),
    ("Marathi", "मराठी", "marathi"), ("Maharashtra", "Pune", "Mumbai", "Nagpur", "Kolhapur")))
_add(Language("ne", "Nepali", "नेपाली", ("devanagari",),
    _sw("छ छन् छु हो होइन र मा को का की ले लाई बाट पनि तर गर्छ गर्छन् गर्यो गरेको भएको थियो थिए हुन्छ हुन् म हामी तपाईं तिमी उ उनी यो त्यो यस्तो कस्तो किन कति धेरै सबै भन्ने भने अनि छैन गर्नु भयो हुनुहुन्छ"),
    ("Nepali", "नेपाली", "nepali"), ("Sikkim", "Darjeeling", "Nepal", "Kathmandu")))
_add(Language("sa", "Sanskrit", "संस्कृतम्", ("devanagari",),
    _sw("अस्ति सन्ति च तत् एव इति न सः सा अपि भवति अहम् त्वम् वयम् यत् किम् कथम् अत्र तत्र इदम् एतत् तस्य तस्याः तेषाम् नमः एवम् इदानीम् भवान् भवती अस्मि असि स्म कुत्र यदा तदा"),
    ("Sanskrit", "संस्कृत", "sanskrit", "samskritam"), ("Varanasi", "Bengaluru", "Kerala")))
_add(Language("mai", "Maithili", "मैथिली", ("devanagari",),
    _sw("अछि छथि छी छै हम अहाँ हमर अहाँक केर सँ लेल आ ओ ई नहि नञि भेल छल छलाह करैत कहलनि जे से हुनक अपन कोनो किछु"),
    ("Maithili", "मैथिली", "maithili"), ("Darbhanga", "Madhubani", "Mithila", "Bihar")))
_add(Language("bho", "Bhojpuri", "भोजपुरी", ("devanagari",),
    _sw("बा बानी बाड़े बाड़ी बाटे हवे रहल रहे रहलीं आ ना ओकर हमरा हमार तोहार तोहरा ओकरा ऊ ई कइल कइले करत जाला जाई होखे होई बाकिर बतावल कहल लोग रउआ रउरा"),
    ("Bhojpuri", "भोजपुरी", "bhojpuri"), ("Patna", "Gorakhpur", "Varanasi", "Chhapra")))
_add(Language("kok", "Konkani", "कोंकणी", ("devanagari", "kannada"),
    _sw("आसा आसात म्हजो म्हजी तुजो तुजी ना आनी हांव तूं तो ती तें आमी तुमी हे ही तशें अशें कित्याक कशें जाता जालो करता केलो खंय"),
    ("Konkani", "कोंकणी", "konkani"), ("Goa", "Mangalore", "Karwar")))
_add(Language("doi", "Dogri", "डोगरी", ("devanagari",),
    _sw("ऐ दा दी दे हा हे ने ते ओह इस उस असें तुस तुसें कन्ने बी नेईं हो जे कि सारे"),
    ("Dogri", "डोगरी", "dogri"), ("Jammu",)))
_add(Language("brx", "Bodo", "बड़ो", ("devanagari",),
    _sw("आं नों बे बै नङा जों दं गैया मानो बिसोर बिनि जानाय खालाम गोनां बेयो नोंथां आंनि नोंनि"),
    ("Bodo", "बड़ो", "bodo", "boro"), ("Kokrajhar", "Assam")))
_add(Language("awa", "Awadhi", "अवधी", ("devanagari",),
    _sw("हय अहै रहा रहै कै म अउर उ करत कीन कहेन बाटै हमका तोहका उनका"),
    ("Awadhi", "अवधी", "awadhi"), ("Lucknow", "Ayodhya", "Faizabad")))
_add(Language("mag", "Magahi", "मगही", ("devanagari",),
    _sw("हल हई हम्मर तोहर हमनी तोहनी ओकर ओकरा आउ हलई हलथिन"),
    ("Magahi", "मगही", "magahi"), ("Gaya", "Patna", "Nalanda")))
_add(Language("hne", "Chhattisgarhi", "छत्तीसगढ़ी", ("devanagari",),
    _sw("हवय ल ह बर संग अउ नइ नइये करथे करिस होगे रिहिस हमन तुमन ओमन"),
    ("Chhattisgarhi", "छत्तीसगढ़ी", "chhattisgarhi"), ("Raipur", "Bilaspur", "Chhattisgarh")))
_add(Language("raj", "Rajasthani", "राजस्थानी", ("devanagari",),
    _sw("छै छे नै री रो रा म्हारो थारो म्हे थे कोनी कर्यो गयो जावै आवै म्हारा थारा बात"),
    ("Rajasthani", "Marwari", "राजस्थानी", "मारवाड़ी", "rajasthani", "marwari"), ("Jodhpur", "Jaipur", "Bikaner", "Udaipur")))
_add(Language("bgc", "Haryanvi", "हरियाणवी", ("devanagari",),
    _sw("सै सो तै म्हारा थारा कोनी करया आवै जावै तन्ने मन्ने"),
    ("Haryanvi", "हरियाणवी", "haryanvi"), ("Rohtak", "Hisar", "Haryana")))
_add(Language("gbm", "Garhwali", "गढ़वळि", ("devanagari",),
    _sw("छन च छौं मि त्वे मेरु तेरु नि"),
    ("Garhwali", "गढ़वाली", "garhwali"), ("Dehradun", "Pauri", "Uttarakhand")))
_add(Language("kfy", "Kumaoni", "कुमाऊँनी", ("devanagari",),
    _sw("छु त्वील म्यर त्यर नि"),
    ("Kumaoni", "कुमाऊँनी", "kumaoni"), ("Nainital", "Almora", "Uttarakhand")))

_add(Language("bn", "Bengali", "বাংলা", ("bengali",),
    _sw("এবং আমি তুমি আপনি সে তারা আমরা করে হয় হয়েছে না এই সেই আছে ছিল হবে কি কেন কিভাবে জন্য থেকে দিয়ে কিন্তু তবে যে যা যদি তাহলে খুব সব আর ও তো এটা ওটা আমার তোমার আপনার"),
    ("Bengali", "Bangla", "বাংলা", "bengali", "bangla"), ("Kolkata", "West Bengal", "Tripura", "Bangladesh")))
_add(Language("as", "Assamese", "অসমীয়া", ("bengali",),
    _sw("আৰু মই তুমি আপুনি তেওঁ তেওঁলোক আমি কৰে হয় হৈছে নাই এই সেই আছে আছিল হ'ব কি কিয় কেনেকৈ বাবে পৰা দি কিন্তু যে যি যদি তেন্তে বহুত সকলো মোৰ তোমাৰ আপোনাৰ"),
    ("Assamese", "অসমীয়া", "assamese", "axomiya"), ("Guwahati", "Assam", "Dibrugarh")))
_add(Language("mni", "Manipuri", "ꯃꯤꯇꯩꯂꯣꯟ", ("meetei_mayek", "bengali"),
    _sw("ꯑꯃꯁꯨꯡ ꯑꯩ ꯅꯪ ꯃꯍꯥꯛ ꯑꯩꯈꯣꯌ ꯂꯩ ꯑꯣꯏ ꯅꯠꯇꯦ"),
    ("Manipuri", "Meitei", "Meiteilon", "manipuri", "meitei"), ("Imphal", "Manipur")))

_add(Language("pa", "Punjabi", "ਪੰਜਾਬੀ", ("gurmukhi",), _sw(""), ("Punjabi", "ਪੰਜਾਬੀ", "punjabi"), ("Punjab", "Amritsar", "Ludhiana", "Chandigarh")))
_add(Language("gu", "Gujarati", "ગુજરાતી", ("gujarati",), _sw(""), ("Gujarati", "ગુજરાતી", "gujarati"), ("Gujarat", "Ahmedabad", "Surat", "Rajkot")))
_add(Language("or", "Odia", "ଓଡ଼ିଆ", ("oriya",), _sw(""), ("Odia", "Oriya", "ଓଡ଼ିଆ", "odia"), ("Odisha", "Bhubaneswar", "Cuttack")))
_add(Language("ta", "Tamil", "தமிழ்", ("tamil",), _sw(""), ("Tamil", "தமிழ்", "tamil"), ("Tamil Nadu", "Chennai", "Madurai", "Coimbatore")))
_add(Language("te", "Telugu", "తెలుగు", ("telugu",), _sw(""), ("Telugu", "తెలుగు", "telugu"), ("Andhra Pradesh", "Telangana", "Hyderabad", "Vijayawada")))
_add(Language("kn", "Kannada", "ಕನ್ನಡ", ("kannada",),
    _sw("ಮತ್ತು ಇದು ಅದು ನಾನು ನೀವು ಅವರು ನಾವು ಇದೆ ಇಲ್ಲ ಆಗಿದೆ ಎಂದು ಒಂದು ಹಾಗೂ ಆದರೆ ಮಾಡಿ ಮಾಡುತ್ತಾರೆ ನನ್ನ ನಿಮ್ಮ ಅವರ ಏನು ಯಾಕೆ ಹೇಗೆ"),
    ("Kannada", "ಕನ್ನಡ", "kannada"), ("Karnataka", "Bengaluru", "Mysuru", "Hubli")))
_add(Language("tcy", "Tulu", "ತುಳು", ("kannada",),
    _sw("ಉಂಡು ಇಜ್ಜಿ ಎಂಕ್ ಈರ್ ಯಾನ್ ಆಯೆ ಆಳ್ ಅಕುಲು ಎಂಕುಲು ಪಂಡ್ ಬೊಕ್ಕ ಮಲ್ಪುನ ಎಂಚ ದಾನೆ"),
    ("Tulu", "ತುಳು", "tulu"), ("Mangalore", "Udupi")))
_add(Language("ml", "Malayalam", "മലയാളം", ("malayalam",), _sw(""), ("Malayalam", "മലയാളം", "malayalam"), ("Kerala", "Kochi", "Thiruvananthapuram", "Kozhikode")))

_add(Language("ur", "Urdu", "اردو", ("arabic",),
    _sw("ہے ہیں کا کی کے میں کو سے اور نہیں تھا تھی تھے یہ وہ ہم آپ تم پر بھی کہ جو تو اب کیا کیسے لیے گیا گئی گئے ہو ہوں ان اس"),
    ("Urdu", "اردو", "urdu"), ("Lucknow", "Hyderabad", "Delhi"), asr_supported=False))
_add(Language("sd", "Sindhi", "سنڌي", ("arabic", "devanagari"),
    _sw("آهي آهن جو جي جا ۾ کي کان ۽ نه هو هئا هي اسان توهان ڪري ٿو ٿي ٿا ڇا ڪيئن لاءِ مان"),
    ("Sindhi", "سنڌي", "सिंधी", "sindhi"), ("Kutch", "Ulhasnagar", "Ahmedabad")))
_add(Language("ks", "Kashmiri", "کٲشُر", ("arabic", "devanagari"),
    _sw("چھُ چھِ ہس منز تہ سُ اسہ توہہ کیازِ"),
    ("Kashmiri", "کٲشُر", "kashmiri"), ("Srinagar", "Kashmir"), asr_supported=False))

_add(Language("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", ("ol_chiki",), _sw(""), ("Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "santali", "santhali"), ("Jharkhand", "Odisha", "West Bengal")))
_add(Language("en", "English", "Indian English", ("latin",),
    _sw("the a an and or of to in is are was were be been it this that with for on as at by from not you i we they he she"),
    ("Indian English",), ("Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"),
    query_hint=(
        "INDIAN ENGLISH ONLY. Every query must target content spoken by Indians in India: Indian podcasters, founders, "
        "professors (IIT, IIM, IISc, AIIMS, DU), civil servants (IAS/IPS, UPSC preparation), doctors, lawyers (Supreme Court, "
        "High Court), chartered accountants, cricketers, journalists of Indian English news channels, ISRO/DRDO scientists, "
        "Indian authors, Indian-startup ecosystem, NEET/JEE/CAT coaching in English, Indian history and policy talks. Anchor "
        "each query with explicit Indian markers (India, Indian, Bharat, city names, institutions, rupees, Lok Sabha, NITI Aayog, "
        "Nifty, RBI, SEBI). NEVER target American, British, Australian or generic international creators, and never Indian-"
        "diaspora-abroad content. Plain-English topics without an Indian anchor are forbidden.")))
_add(Language("lus", "Mizo", "Mizo ṭawng", ("latin",),
    _sw("a leh chu hi nge te an kan in ka i lo tih chuan"),
    ("Mizo", "mizo"), ("Aizawl", "Mizoram")))
_add(Language("grt", "Garo", "A·chik", ("latin",),
    _sw("ang na·a ua nang ia gita ba aro ong·a ong·jok"),
    ("Garo", "garo", "achik"), ("Tura", "Meghalaya")))
_add(Language("kha", "Khasi", "Khasi", ("latin",),
    _sw("ka u ki ba bad jong na ha sha kumno kumne dei"),
    ("Khasi", "khasi"), ("Shillong", "Meghalaya")))


# Script -> candidate languages (order = default prior when nothing else discriminates)
SCRIPT_LANGS: dict[str, tuple[str, ...]] = {
    "devanagari": ("hi", "mr", "ne", "sa", "mai", "bho", "kok", "doi", "brx", "awa", "mag", "hne", "raj", "bgc", "gbm", "kfy", "sd"),
    "bengali": ("bn", "as", "mni"),
    "gurmukhi": ("pa",),
    "gujarati": ("gu",),
    "oriya": ("or",),
    "tamil": ("ta",),
    "telugu": ("te",),
    "kannada": ("kn", "tcy", "kok"),
    "malayalam": ("ml",),
    "arabic": ("ur", "sd", "ks"),
    "ol_chiki": ("sat",),
    "meetei_mayek": ("mni",),
    "latin": ("en", "lus", "grt", "kha"),
}

SCRIPT_NAMES: dict[str, str] = {
    "devanagari": "Devanagari", "bengali": "Bengali-Assamese", "gurmukhi": "Gurmukhi", "gujarati": "Gujarati",
    "oriya": "Odia", "tamil": "Tamil", "telugu": "Telugu", "kannada": "Kannada", "malayalam": "Malayalam",
    "arabic": "Perso-Arabic", "ol_chiki": "Ol Chiki", "meetei_mayek": "Meetei Mayek", "latin": "Latin",
}

# Content domains to sweep so the corpus is topically broad (the LLM rotates through these).
DISCOVERY_DOMAINS: tuple[str, ...] = (
    "education and exam preparation", "school and university lectures", "personal finance and investing", "stock market and economy",
    "banking, insurance and loans", "business, startups and entrepreneurship", "sports analysis (cricket, football, kabaddi, hockey)",
    "news analysis and current affairs", "politics and elections", "government schemes and policy", "law, courts and citizen rights",
    "health, medicine and doctors' advice", "mental health and psychology", "fitness, yoga and nutrition", "science and technology explainers",
    "AI, software and gadgets", "agriculture, farming and animal husbandry", "rural life and village stories", "history and heritage",
    "geography and travel", "religion, spirituality and philosophy", "mythology and epics narration", "literature, poetry and book discussion",
    "cinema and music industry talk (spoken discussion only)", "food, cooking and nutrition talk", "environment, climate and wildlife",
    "career guidance and job skills", "parenting, family and relationships", "women's issues and social change", "motivation and self improvement",
    "language learning and grammar", "folk tales and moral stories", "biographies of leaders, scientists and artists", "automobiles and real estate",
    "astronomy and space", "local culture, festivals and traditions", "public speaking and debate", "interviews with professionals and artisans",
    "true stories and crime narration", "audiobooks and long-form narration",
)

# Formats that tend to be clean, single-speaker, spoken-word audio.
DISCOVERY_GENRES: tuple[str, ...] = (
    "long form podcast", "one on one interview", "audiobook narration", "story narration", "motivational speech",
    "university lecture", "news analysis monologue", "history documentary narration", "biography narration",
    "self improvement talk", "spiritual discourse", "science explainer", "book summary", "radio talk show",
    "poetry recitation", "moral stories for adults", "financial literacy explainer", "health talk by doctor",
    "farming advice talk", "travel vlog commentary", "cooking recipe narration", "technology explainer",
    "law explainer", "career guidance talk", "mythology narration", "philosophy talk", "language learning lesson",
    "current affairs commentary", "sports commentary analysis", "parenting advice talk",
)


def get(code: str) -> Language:
    return LANGUAGES[code]
