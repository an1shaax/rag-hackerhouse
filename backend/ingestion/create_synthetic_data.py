#!/usr/bin/env python3
"""
Create synthetic MSMARCO-XI-like data for development and testing.

The full dataset is 55GB and takes too long to download during development.
This script creates a smaller synthetic dataset with the same schema
that can be used for building and testing the pipeline.

Usage:
    python create_synthetic_data.py --num-records 5000 --output-dir ../data
"""
import argparse
import json
import random
from pathlib import Path


# Realistic queries and answers for multiple languages
SAMPLE_DATA = [
    {
        "query_en": "What is the capital of India?",
        "answer_en": "New Delhi is the capital of India. It was officially declared the capital of India in 1911 by the British. New Delhi is located in the National Capital Territory of Delhi and serves as the seat of all three branches of the Government of India.",
        "query_hi": "भारत की राजधानी क्या है?",
        "answer_hi": "नई दिल्ली भारत की राजधानी है। इसे 1911 में ब्रिटिशों द्वारा भारत की राजधानी घोषित किया गया था। नई दिल्ली दिल्ली के राष्ट्रीय राजधानी क्षेत्र में स्थित है।",
        "query_bn": "ভারতের রাজধানী কী?",
        "answer_bn": "নতুন দিল্লি ভারতের রাজধানী। ১৯১১ সালে ব্রিটিশরা এটিকে ভারতের রাজধানী ঘোষণা করেন। নতুন দিল্লি দিল্লির জাতীয় রাজধানী অঞ্চলে অবস্থিত।",
        "query_ta": "இந்தியாவின் தலைநகரம் என்ன?",
        "answer_ta": "புது டெல்லி இந்தியாவின் தலைநகரம் ஆகும். 1911 இல் பிரிட்டிஷார் இதை இந்தியாவின் தலைநகரமாக அறிவித்தனர்.",
        "query_te": "భారతదేశం యొక్క రాజధాని ఏమిటి?",
        "answer_te": "న్యూఢిల్లీ భారతదేశ రాజధాని. 1911లో బ్రిటిష్ వారు దీనిని భారతదేశ రాజధానిగా ప్రకటించారు.",
        "passages": [
            "New Delhi is the capital of India and the seat of all three branches of the Government of India.",
            "The National Capital Territory of Delhi covers an area of 1,484 square kilometres.",
            "New Delhi was designed by British architects Edwin Lutyens and Herbert Baker.",
            "The Raisina Hill area houses the important government buildings including Rashtrapati Bhavan.",
        ],
        "query_type": "entity",
    },
    {
        "query_en": "How does photosynthesis work?",
        "answer_en": "Photosynthesis is the process by which plants convert light energy into chemical energy. It takes place in the chloroplasts of plant cells, specifically in the thylakoid membranes. The process involves two main stages: the light-dependent reactions and the Calvin cycle (light-independent reactions). Chlorophyll absorbs sunlight, water is split, and carbon dioxide is converted to glucose.",
        "query_hi": "प्रकाश संश्लेषण कैसे काम करता है?",
        "answer_hi": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं। यह पौधों की कोशिकाओं के क्लोरोप्लास्ट में होता है। इसमें दो मुख्य चरण होते हैं: प्रकाश-निर्भर प्रतिक्रियाएं और कैल्विन चक्र।",
        "passages": [
            "Photosynthesis occurs in the chloroplasts of plant cells.",
            "The light-dependent reactions take place in the thylakoid membranes.",
            "The Calvin cycle converts CO2 into glucose using ATP and NADPH.",
            "Chlorophyll is the primary pigment that absorbs light energy.",
        ],
        "query_type": "definition",
    },
    {
        "query_en": "What is machine learning?",
        "answer_en": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning. Common algorithms include neural networks, decision trees, support vector machines, and random forests.",
        "query_hi": "मशीन लर्निंग क्या है?",
        "answer_hi": "मशीन लर्निंग एक कृत्रिम बुद्धिमत्ता का उपसमूह है जो कंप्यूटर को डेटा से सीखने और निर्णय लेने में सक्षम बनाता है। इसमें सुपरवाइज्ड लर्निंग, अनसुपरवाइज्ड लर्निंग और रीइन्फोर्समेंट लर्निंग शामिल है।",
        "passages": [
            "Machine learning is a subset of artificial intelligence (AI).",
            "It enables systems to learn from data and improve over time.",
            "Common types include supervised, unsupervised, and reinforcement learning.",
            "Neural networks, decision trees, and SVMs are popular algorithms.",
        ],
        "query_type": "definition",
    },
    {
        "query_en": "What are the major rivers of India?",
        "answer_en": "India has several major rivers. The Ganges is the longest river in India at 2,525 km. The Godavari is the second longest at 1,465 km. The Krishna River flows 1,400 km. Other important rivers include the Yamuna, Narmada, Brahmaputra, and Mahanadi.",
        "query_hi": "भारत की प्रमुख नदियाँ कौन सी हैं?",
        "answer_hi": "भारत में कई प्रमुख नदियाँ हैं। गंगा भारत की सबसे लंबी नदी है जो 2,525 किमी लंबी है। गोदावरी दूसरी सबसे लंबी नदी है। अन्य महत्वपूर्ण नदियों में यमुना, नर्मदा, ब्रह्मपुत्र और महानदी शामिल हैं।",
        "passages": [
            "The Ganges is the longest river in India at 2,525 kilometres.",
            "The Godavari River is the second longest river at 1,465 km.",
            "The Yamuna River is a major tributary of the Ganges.",
            "The Brahmaputra River flows through northeastern India.",
        ],
        "query_type": "description",
    },
    {
        "query_en": "Who invented the telephone?",
        "answer_en": "Alexander Graham Bell is credited with inventing the telephone. He was awarded the first US patent for the invention in 1876. However, there is controversy as Antonio Meucci also developed a voice-communicating device earlier. Elisha Gray independently filed a patent caveat on the same day as Bell.",
        "query_hi": "टेलीफोन का आविष्कार किसने किया?",
        "answer_hi": "अलेक्जेंडर ग्राहम बेल को टेलीफोन के आविष्कारक के रूप में जाना जाता है। उन्हें 1876 में इसके लिए पहला अमेरिकी पेटेंट मिला। हालांकि, एंटोनियो मेउची ने भी पहले एक आवाज़ संचार डिवाइस विकसित किया था।",
        "passages": [
            "Alexander Graham Bell invented the telephone in 1876.",
            "He received the first US patent for the telephone.",
            "Antonio Meucci developed an earlier voice-communicating device.",
            "Elisha Gray filed a patent caveat on the same day as Bell.",
        ],
        "query_type": "entity",
    },
    {
        "query_en": "What is the solar system?",
        "answer_en": "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It consists of eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The four inner planets are rocky (terrestrial) and the four outer planets are gas or ice giants. The Solar System also includes dwarf planets, asteroids, and comets.",
        "query_hi": "सौर मंडल क्या है?",
        "answer_hi": "सौर मंडल सूर्य और उसके चारों ओर परिक्रमा करने वाली वस्तुओं का गुरुत्वाकर्षण बंध प्रणाली है। इसमें आठ ग्रह हैं: बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, अरुण और वरुण।",
        "passages": [
            "The Solar System consists of eight planets orbiting the Sun.",
            "Mercury, Venus, Earth, and Mars are inner rocky planets.",
            "Jupiter, Saturn, Uranus, and Neptune are outer gas/ice giants.",
            "The Solar System also includes dwarf planets, asteroids, and comets.",
        ],
        "query_type": "definition",
    },
    {
        "query_en": "What is climate change?",
        "answer_en": "Climate change refers to long-term shifts in temperatures and weather patterns. Human activities, primarily the burning of fossil fuels, have been the main driver of climate change since the 1800s. This leads to rising global temperatures, melting ice caps, rising sea levels, and more extreme weather events.",
        "query_hi": "जलवायु परिवर्तन क्या है?",
        "answer_hi": "जलवायु परिवर्तन तापमान और मौसम के पैटर्न में दीर्घकालिक बदलावों को संदर्भित करता है। मानवीय गतिविधियाँ, विशेष रूप से जीवाश्म ईंधन का जलना, 1800 के दशक से जलवायु परिवर्तन का मुख्य कारण रही हैं।",
        "passages": [
            "Climate change refers to long-term shifts in temperatures and weather patterns.",
            "Human activities are the main driver since the 1800s.",
            "Effects include rising temperatures, melting ice, and rising sea levels.",
            "Burning fossil fuels releases greenhouse gases that trap heat.",
        ],
        "query_type": "definition",
    },
    {
        "query_en": "What are the IT hubs of India?",
        "answer_en": "India's major IT hubs include Bengaluru (Bangalore) which is known as the Silicon Valley of India, Hyderabad, Pune, Chennai, Mumbai, Noida, and Gurugram (Gurugram). Bengaluru alone accounts for about 35% of India's IT exports. These cities have thriving technology ecosystems with major companies like TCS, Infosys, and Wipro.",
        "query_hi": "भारत के आईटी हब कौन से हैं?",
        "answer_hi": "भारत के प्रमुख आईटी हब में बेंगलुरु (बैंगलोर), हैदराबाद, पुणे, चेन्नई, मुंबई, नोएडा और गुरुग्राम शामिल हैं। बेंगलुरु भारत का सिलिकॉन वैली के नाम से जाना जाता है।",
        "passages": [
            "Bengaluru is known as the Silicon Valley of India.",
            "Major IT hubs include Hyderabad, Pune, Chennai, and Mumbai.",
            "Bengaluru accounts for about 35% of India's IT exports.",
            "TCS, Infosys, and Wipro are major Indian IT companies.",
        ],
        "query_type": "description",
    },
    {
        "query_en": "What is blockchain technology?",
        "answer_en": "Blockchain is a decentralized, distributed ledger technology that records transactions across many computers. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. It ensures transparency, immutability, and security. Cryptocurrencies like Bitcoin and Ethereum are built on blockchain technology.",
        "query_hi": "ब्लॉकचेन तकनीक क्या है?",
        "answer_hi": "ब्लॉकचेन एक विकेंद्रीकृत, वितरित लेजर तकनीक है जो कई कंप्यूटरों में लेनदेन रिकॉर्ड करती है। यह पारदर्शिता, अपरिवर्तनीयता और सुरक्षा सुनिश्चित करता है।",
        "passages": [
            "Blockchain is a decentralized, distributed ledger technology.",
            "Each block contains a cryptographic hash, timestamp, and transaction data.",
            "It ensures transparency, immutability, and security.",
            "Bitcoin and Ethereum are built on blockchain technology.",
        ],
        "query_type": "definition",
    },
    {
        "query_en": "What is the Indian constitution?",
        "answer_en": "The Constitution of India is the supreme law of India. It was adopted on 26 January 1950 and is the longest written constitution of any sovereign country. Dr. B.R. Ambedkar is known as the Father of the Indian Constitution. It establishes the framework of the Indian political system and guarantees fundamental rights to all citizens.",
        "query_hi": "भारत का संविधान क्या है?",
        "answer_hi": "भारत का संविधान भारत का सर्वोच्च कानून है। इसे 26 जनवरी 1950 को अपनाया गया था। डॉ. बी.आर. अंबेडकर को भारतीय संविधान के जनक के रूप में जाना जाता है।",
        "passages": [
            "The Constitution of India is the supreme law of India.",
            "It was adopted on 26 January 1950.",
            "Dr. B.R. Ambedkar is known as the Father of the Indian Constitution.",
            "It is the longest written constitution of any sovereign country.",
        ],
        "query_type": "description",
    },
    {
        "query_en": "What is the capital of Australia?",
        "answer_en": "Canberra is the capital city of Australia. It was selected as the capital in 1908 as a compromise between Sydney and Melbourne. Canberra is located in the Australian Capital Territory (ACT) and is home to Parliament House, the High Court, and many government departments.",
        "query_hi": "ऑस्ट्रेलिया की राजधानी क्या है?",
        "answer_hi": "कैनबरा ऑस्ट्रेलिया की राजधानी है। इसे 1908 में सिडनी और मेलबर्न के बीच समझौते के रूप में राजधानी चुना गया था। कैनबरा ऑस्ट्रेलियाई राजधानी क्षेत्र (एसीटी) में स्थित है।",
        "passages": [
            "Canberra is the capital city of Australia.",
            "It was selected as the capital in 1908 as a compromise between Sydney and Melbourne.",
            "Canberra is located in the Australian Capital Territory (ACT).",
            "Parliament House and the High Court are located in Canberra.",
        ],
        "query_type": "entity",
    },
]

def generate_dataset(num_records: int = 5000, output_dir: str = "data"):
    """Generate synthetic dataset in the MSMARCO-XI format"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = []

    for i in range(num_records):
        base = SAMPLE_DATA[i % len(SAMPLE_DATA)]
        lang_codes = ["en", "hi", "bn", "ta", "te"]
        lang = random.choice(lang_codes)
        lang_map = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu"}

        query = base.get(f"query_{lang}", base["query_en"])
        answer = base.get(f"answer_{lang}", base["answer_en"])

        # Create variations
        if i >= len(SAMPLE_DATA):
            suffixes = ["", " in detail", " briefly", " for students", " with examples"]
            query = query.rstrip(".") + random.choice(suffixes) + "."

        # Select random passages
        selected_passages = random.sample(
            base["passages"],
            k=min(random.randint(1, 3), len(base["passages"]))
        )

        record = {
            "query": query,
            "Answer": answer,
            "Eng_Query": base["query_en"],
            "Eng_Answer": base["answer_en"],
            "query_id": 100000 + i,
            "query_type": base["query_type"],
            "source_lang": "eng_Latn",
            "target_lang": lang_map[lang],
            "passages": {
                "English_passages": base["passages"],
                "Translated_passages": selected_passages,
                "is_selected": [1 if p in selected_passages else 0 for p in base["passages"]],
            },
            "meta": {
                "model_name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
        }

        records.append(record)

    # Save as JSON for easy loading
    output_file = output_path / "synthetic_msmarco_xi.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {num_records} synthetic records")
    print(f"   Output: {output_file}")
    print(f"   Languages: en, hi, bn, ta, te")
    print(f"   Size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    return records


def main():
    parser = argparse.ArgumentParser(description="Create synthetic MSMARCO-XI data")
    parser.add_argument("--num-records", type=int, default=5000, help="Number of records")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()
    generate_dataset(num_records=args.num_records, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
