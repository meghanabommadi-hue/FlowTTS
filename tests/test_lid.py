from chaashini.lid import identify


def test_hindi():
    r = identify("यह एक बहुत अच्छा दिन है और हम सब खुश हैं")
    assert r.lang == "hi" and r.confidence > 0.6


def test_marathi():
    r = identify("मी आज खूप आनंदी आहे आणि आम्ही सगळे बाहेर जाणार आहोत")
    assert r.lang == "mr"


def test_bengali_vs_assamese():
    assert identify("আমি আজ খুব খুশি এবং আমরা সবাই বাইরে যাব").lang == "bn"
    assert identify("মই আজি বহুত সুখী আৰু আমি সকলোৱে বাহিৰলৈ যাম").lang == "as"


def test_single_script():
    assert identify("நான் இன்று மிகவும் மகிழ்ச்சியாக இருக்கிறேன்").lang == "ta"
    assert identify("నేను ఈ రోజు చాలా సంతోషంగా ఉన్నాను").lang == "te"
    assert identify("ਮੈਂ ਅੱਜ ਬਹੁਤ ਖੁਸ਼ ਹਾਂ").lang == "pa"


def test_code_mix():
    r = identify("आज हम discuss करेंगे machine learning के बारे में और उसके applications")
    assert r.lang == "hi" and r.code_mixed and "en" in r.composition


def test_expected_prior():
    r = identify("अच्छा", expected="mr")
    assert r.lang == "mr"
