from flask import Flask, request, render_template, jsonify
import os
import io

# --- Original Imports for Image Classification and Chat ---
try:
    from class_labels import class_names
except ImportError:
    print("Warning: class_labels.py not found. Using default class_names for image classification.")
    class_names = ["Default Artifact"] # Placeholder

try:
    from llm_utils import generate_chat_response
except ImportError:
    print("Warning: llm_utils.py not found. Chat functionality will be a placeholder.")
    def generate_chat_response(user_message, artifact_name, artifact_description):
        return "Chat response generation is currently unavailable due to missing llm_utils."

from keras.saving import load_model
import numpy as np
from PIL import Image

# --- New Imports for Recommendation Logic ---
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- New Recommendation Logic Setup ---
ATTRACTIONS_DATA = [
    {
        "name": "Great Pyramids of Giza",
        "city": "Giza",
        "description": "The last remaining wonder of the ancient world, massive structures built as tombs for the pharaohs.",
        "type": "Pharaonic",
        "popularity": 10,
        "period": "Old Kingdom",
        "dynasty": "4th Dynasty",
        "notable_pharaohs": ["Khufu", "Khafre", "Menkaure"],
        "significance": "Largest pyramids ever built, demonstrate advanced engineering and astronomical alignment"
    },
    {
        "name": "Egyptian Museum",
        "city": "Cairo",
        "description": "Home to the world's largest collection of Pharaonic antiquities, including treasures from Tutankhamun's tomb.",
        "type": "Pharaonic",
        "popularity": 9,
        "key_artifacts": ["Tutankhamun's Death Mask", "Royal Mummies Collection", "Narmer Palette", "Statue of Khufu"]
    },
    {
        "name": "Karnak Temple",
        "city": "Luxor",
        "description": "A vast temple complex dedicated to the Theban triad of Amun, Mut, and Khonsu, featuring massive columns and obelisks.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom to Ptolemaic",
        "notable_features": ["Great Hypostyle Hall", "Sacred Lake", "Avenue of Sphinxes", "Obelisks of Hatshepsut"],
        "significance": "Largest religious building ever constructed, built over 2000 years"
    },
    {
        "name": "Valley of the Kings",
        "city": "Luxor",
        "description": "Royal burial ground containing tombs of pharaohs from the New Kingdom, including Tutankhamun.",
        "type": "Pharaonic",
        "popularity": 9,
        "period": "New Kingdom",
        "dynasty": "18th-20th Dynasties",
        "notable_tombs": ["KV62 (Tutankhamun)", "KV17 (Seti I)", "KV7 (Ramses II)", "KV5 (Sons of Ramses II)"],
        "significance": "Contains 63 tombs with elaborate wall paintings depicting Egyptian mythology"
    },
    {
        "name": "Abu Simbel",
        "city": "Aswan",
        "description": "Massive rock temples built by Ramses II, featuring colossal statues and intricate carvings.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "history": "Relocated in 1968 to save it from the rising waters of Lake Nasser",
        "significance": "Shows deification of Ramses II and architectural innovation"
    },

    # Existing Pharaonic sites with expanded information
    {
        "name": "Luxor Temple",
        "city": "Luxor",
        "description": "Ancient Egyptian temple complex located on the east bank of the Nile River, known for its colossal statues and beautiful colonnades.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom",
        "dynasty": "18th-19th Dynasty",
        "notable_pharaohs": ["Amenhotep III", "Ramses II"],
        "significance": "Connected to Karnak by the Avenue of Sphinxes, dedicated to rejuvenation of kingship"
    },
    {
        "name": "Temple of Hatshepsut",
        "city": "Luxor",
        "description": "Mortuary temple of the female pharaoh Hatshepsut, featuring terraced colonnades set against dramatic cliffs.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "significance": "Innovative architectural design, important female pharaoh's monument",
        "history": "Many images of Hatshepsut were destroyed after her death by her successor"
    },
    {
        "name": "Step Pyramid of Djoser",
        "city": "Saqqara",
        "description": "The world's oldest major stone structure, built in the 27th century BC as a tomb for Pharaoh Djoser.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "Old Kingdom",
        "dynasty": "3rd Dynasty",
        "architect": "Imhotep",
        "significance": "First pyramid ever built, revolutionary use of stone architecture"
    },
    {
        "name": "Philae Temple",
        "city": "Aswan",
        "description": "Island temple complex dedicated to the goddess Isis, rescued from the rising waters of Lake Nasser after the Aswan Dam.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Ptolemaic to Roman",
        "significance": "Last active temple of the ancient Egyptian religion, hieroglyphics were still being added in the 5th century AD"
    },

    # New additional Ancient Egyptian sites
    {
        "name": "Tomb of Nefertari",
        "city": "Luxor",
        "description": "The most beautifully decorated tomb in the Valley of the Queens, belonging to the favorite wife of Ramses II.",
        "type": "Pharaonic",
        "popularity": 9,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "significance": "Often called 'The Sistine Chapel of Ancient Egypt' for its vivid paintings",
        "limited_access": True,
        "preservation_status": "Carefully preserved with limited daily visitors"
    },
    {
        "name": "Tomb of Seti I",
        "city": "Luxor",
        "description": "The longest and deepest tomb in the Valley of the Kings with exquisite well-preserved reliefs.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "significance": "Contains the complete 'Book of Gates' funerary text with astronomical ceiling",
        "limited_access": True
    },
    {
        "name": "Temple of Hathor at Dendera",
        "city": "Qena",
        "description": "Well-preserved temple complex dedicated to the goddess Hathor, featuring one of the most intact temple buildings in Egypt.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Ptolemaic to Roman",
        "notable_features": ["Famous Dendera Zodiac (now in Louvre)", "Crypts with mysterious reliefs", "Roof chapels"],
        "significance": "Contains the controversial 'Dendera Light' relief some interpret as depicting ancient electricity"
    },
    {
        "name": "Medinet Habu",
        "city": "Luxor",
        "description": "Mortuary temple of Ramses III with exceptionally well-preserved colorful reliefs depicting religious rituals and wars.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "dynasty": "20th Dynasty",
        "significance": "Contains important records of the Sea Peoples invasions",
        "notable_features": ["Migdol Gate", "Calendar of Feasts", "Royal Palace ruins"]
    },
    {
        "name": "Temple of Kom Ombo",
        "city": "Kom Ombo",
        "description": "Unusual double temple dedicated equally to the crocodile god Sobek and the falcon god Horus.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Ptolemaic to Roman",
        "unique_feature": "Perfectly symmetrical twin temples with two entrances, sanctuaries and dedicated areas",
        "additional_attraction": "Contains ancient medical instruments and a crocodile mummy exhibition"
    },
    {
        "name": "Ramesseum",
        "city": "Luxor",
        "description": "Mortuary temple of Ramses II with massive fallen colossus that inspired Shelley's poem 'Ozymandias'.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "notable_features": ["Fallen 57-foot colossus", "Battle of Kadesh reliefs", "Astronomical ceiling"],
        "cultural_significance": "Inspired Percy Shelley's famous poem 'Ozymandias'"
    },
    {
        "name": "Temple of Edfu",
        "city": "Edfu",
        "description": "One of the best-preserved ancient temples in Egypt, dedicated to the falcon god Horus.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Ptolemaic",
        "significance": "Most complete and best-preserved major temple in Egypt",
        "notable_features": ["Massive pylons", "Sacred barque shrine", "Inscriptions about the conflict between Horus and Seth"]
    },
    {
        "name": "Abydos Temple Complex",
        "city": "Sohag",
        "description": "Ancient sacred site containing the famous Abydos King List and magnificent temple of Seti I with mysterious hieroglyphs.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Old Kingdom to New Kingdom",
        "significance": "One of the most sacred sites in ancient Egypt, associated with Osiris and the afterlife",
        "enigmatic_feature": "Contains the 'Abydos helicopter' hieroglyphs that some claim show modern technology"
    },
    {
        "name": "Tombs of the Nobles",
        "city": "Luxor",
        "description": "Collection of private tombs of high officials and nobles with vivid scenes of daily life in ancient Egypt.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "significance": "Provides unique insights into everyday ancient Egyptian life rather than religious scenes",
        "notable_tombs": ["Tomb of Menna", "Tomb of Nakht", "Tomb of Rekhmire"]
    },
    {
        "name": "Temple of Khnum",
        "city": "Esna",
        "description": "Partially excavated temple dedicated to the ram-headed creator god Khnum who formed humans on his potter's wheel.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Ptolemaic to Roman",
        "unique_feature": "The hypostyle hall sits 9 meters below modern street level",
        "significance": "Contains some of the last hieroglyphic inscriptions ever carved"
    },
    {
        "name": "Tombs of Beni Hasan",
        "city": "Minya",
        "description": "Rock-cut tombs of provincial governors carved into limestone cliffs, featuring unique wrestling scenes and military activities.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Middle Kingdom",
        "dynasty": "11th-12th Dynasties",
        "significance": "Shows provincial art styles different from official royal monuments"
    },
    {
        "name": "Pyramid of Unas",
        "city": "Saqqara",
        "description": "Final pyramid built in the 5th Dynasty, containing the oldest known Pyramid Texts - religious spells to help the king in the afterlife.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Old Kingdom",
        "dynasty": "5th Dynasty",
        "significance": "First pyramid to contain the Pyramid Texts, the oldest religious writings in the world"
    },
    {
        "name": "Pyramids of Dahshur",
        "city": "Dahshur",
        "description": "Royal necropolis with unique experimental pyramids showing the evolution of pyramid construction.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Old Kingdom",
        "notable_structures": ["Red Pyramid", "Bent Pyramid", "Black Pyramid"],
        "significance": "Shows evolution of pyramid design and engineering solutions"
    },
    {
        "name": "Deir el-Medina",
        "city": "Luxor",
        "description": "Ancient village of the artisans who worked on the tombs in the Valley of the Kings, with their own tombs and temples.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "significance": "Provides unprecedented insight into everyday life of ordinary Egyptians",
        "archaeological_importance": "Preserved thousands of ostraca (limestone flakes) with records of daily life"
    },
    {
        "name": "Temple of Hibis",
        "city": "Kharga Oasis",
        "description": "The best-preserved temple in the Western Desert, dedicated to the Theban triad and showing Persian influence.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Late Period to Persian",
        "dynasty": "26th Dynasty to 27th Dynasty",
        "significance": "Rare example of Persian-era temple construction in Egypt"
    },
    {
        "name": "Pyramid Texts at Saqqara",
        "city": "Saqqara",
        "description": "The oldest known religious texts in the world, carved on the walls of the pyramids of Saqqara.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Old Kingdom",
        "dynasty": "5th-6th Dynasties",
        "significance": "Oldest religious corpus in the world, precursor to the Book of the Dead"
    },
    {
        "name": "Obelisk of Senusret I",
        "city": "Heliopolis (Cairo)",
        "description": "The oldest standing obelisk in Egypt, dating from the Middle Kingdom, made of pink granite.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Middle Kingdom",
        "dynasty": "12th Dynasty",
        "significance": "One of the few monuments remaining from ancient Heliopolis sun temple"
    },
    {
        "name": "Tomb of Tuthmosis III",
        "city": "Luxor",
        "description": "The tomb of Egypt's greatest warrior pharaoh, hidden high in the Valley of the Kings cliffs with unique 'stick figure' decorations.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "unique_feature": "Features the earliest complete depiction of the 'Amduat' (Book of What is in the Underworld)"
    },
    {
        "name": "Tanis (San el-Hagar)",
        "city": "Sharqia Governorate",
        "description": "Ancient capital city with royal tombs containing treasures rivaling those of Tutankhamun, featured in Raiders of the Lost Ark.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Third Intermediate Period to Late Period",
        "dynasty": "21st-26th Dynasties",
        "pop_culture": "Featured as the resting place of the Ark of the Covenant in 'Raiders of the Lost Ark'",
        "significance": "Silver coffins and gold funerary masks found intact in royal tombs"
    },
    {
        "name": "Temple of Seti I at Abydos",
        "city": "Sohag",
        "description": "Magnificent temple with finely carved reliefs of exceptional quality and the famous Abydos King List.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "significance": "Contains the Abydos King List, a chronological list of most pharaohs from Menes to Seti I",
        "architectural_feature": "Unique L-shaped design with seven chapels dedicated to different deities"
    },
    {
        "name": "Tomb of Ay",
        "city": "Luxor",
        "description": "Tomb of Tutankhamun's successor in the Western Valley of the Kings, with well-preserved paintings of baboons.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "historical_significance": "Ay was possibly Nefertiti's father and may have been involved in Tutankhamun's death"
    },
    {
        "name": "Tombs of Asasif",
        "city": "Luxor",
        "description": "Massive rock-cut tombs of high officials from the Late Period, some as large as royal tombs with labyrinthine layouts.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Late Period",
        "dynasty": "25th-26th Dynasties",
        "notable_tombs": ["Tomb of Pabasa", "Tomb of Anch-Hor", "Tomb of Pedamenope"],
        "significance": "Shows revival of Old Kingdom artistic styles during the 'Saite Renaissance'"
    },
    {
        "name": "Festival Temple of Thutmose III",
        "city": "Luxor",
        "description": "Temple at Karnak dedicated to the jubilee festivals (Heb-Sed) of Thutmose III, with unique column styles.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "unique_feature": "Features tent-pole shaped columns and unique 'botanical garden' reliefs of foreign plants"
    },
    {
        "name": "The Osireion",
        "city": "Abydos",
        "description": "Mysterious underground structure behind the Temple of Seti I, possibly symbolizing the tomb of Osiris.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "unique_feature": "Subterranean design with massive monolithic blocks similar to Old Kingdom architecture",
        "enigmatic_aspect": "Purpose and symbolism still debated by Egyptologists"
    },
    {
        "name": "Tomb KV5",
        "city": "Luxor",
        "description": "Massive tomb in the Valley of the Kings built for the sons of Ramses II, with over 130 corridors and chambers discovered so far.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "significance": "Largest tomb in the Valley of the Kings, could contain up to 150 burial chambers",
        "discovery": "Rediscovered in modern times by Kent Weeks in 1995"
    },
    {
        "name": "Wadi el-Hudi",
        "city": "Aswan",
        "description": "Ancient amethyst and gold mining region with inscriptions detailing mining expeditions from the Middle Kingdom.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "Middle Kingdom to New Kingdom",
        "archaeological_importance": "Contains inscriptions about mining operations and expedition logistics"
    },
    {
        "name": "Temple of Montu at Medamud",
        "city": "Luxor",
        "description": "Remains of the temple dedicated to the falcon-headed war god Montu, featuring massive columns and a sacred lake.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Middle Kingdom to Ptolemaic",
        "significance": "Celebrated the war god who was important before Amun rose to prominence"
    },
    {
        "name": "Serapeum of Saqqara",
        "city": "Saqqara",
        "description": "Underground galleries containing massive granite sarcophagi for the sacred Apis bulls.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Late Period to Ptolemaic",
        "significance": "Shows the importance of sacred animal cults in late Egyptian religion",
        "enigmatic_feature": "How the ancient Egyptians moved 70-ton sarcophagi through narrow corridors remains a mystery"
    },
    {
        "name": "Temple of Amada",
        "city": "Nubia (Lake Nasser)",
        "description": "Oldest temple remaining in Nubia, relocated to save it from the rising waters of Lake Nasser.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "significance": "Contains important historical texts about Thutmose III's campaigns",
        "preservation": "Relocated in 1964-65 to save it from the Aswan High Dam's waters"
    },
    {
        "name": "Tomb of Meresankh III",
        "city": "Giza",
        "description": "Exceptionally well-preserved tomb of a queen from the 4th Dynasty with vivid colors and statues.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Old Kingdom",
        "dynasty": "4th Dynasty",
        "relationship": "Granddaughter of King Khufu and wife of King Khafre",
        "significance": "One of the best-preserved Old Kingdom tombs with original colors"
    },
    {
        "name": "Temple of Amun at Jebel Barkal",
        "city": "Sudan (Nubia)",
        "description": "Temple at the foot of the 'Holy Mountain' of ancient Nubia, built by Egyptian pharaohs during their control of Kush.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "New Kingdom to Napatan",
        "significance": "UNESCO World Heritage site showing Egyptian cultural influence in Nubia",
        "location_note": "Located in modern-day Sudan, part of ancient Egyptian empire"
    },
    {
        "name": "Mastaba of Ti",
        "city": "Saqqara",
        "description": "Old Kingdom tomb with some of the finest relief carvings showing detailed scenes of daily life.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Old Kingdom",
        "dynasty": "5th Dynasty",
        "significance": "Exceptional quality reliefs showing farming, hunting, crafts and other daily activities",
        "historical_figure": "Ti was Overseer of the Pyramids and Sun Temples during the 5th Dynasty"
    },
    {
        "name": "Lahun Pyramid",
        "city": "Faiyum",
        "description": "Middle Kingdom pyramid of Senusret II built with a unique internal mud-brick structure.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Middle Kingdom",
        "dynasty": "12th Dynasty",
        "significance": "Unique construction technique using mud brick with limestone casing",
        "discovery": "Treasure of Princess Sithathoriunet found nearby in 1914"
    },
    {
        "name": "Necropolis of El-Kab",
        "city": "El-Kab",
        "description": "Ancient walled city with rock-cut tombs containing important biographical inscriptions from Egypt's history.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Old Kingdom to New Kingdom",
        "significance": "Contains the tomb of Ahmose, son of Ebana, with account of expulsion of the Hyksos"
    },
    {
        "name": "Crocodile Mummies of Kom Ombo",
        "city": "Kom Ombo",
        "description": "Exhibition of mummified crocodiles and coffins found near the Temple of Kom Ombo.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Ptolemaic to Roman",
        "significance": "Demonstrates ancient Egyptian animal worship and mummification practices",
        "connection": "Associated with worship of Sobek, the crocodile god"
    },
    {
        "name": "Temple of Ptah at Memphis",
        "city": "Mit Rahina",
        "description": "Remains of the temple dedicated to Ptah, creator god and patron of craftsmen in the ancient capital of Memphis.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Old Kingdom to New Kingdom",
        "significance": "Main temple in Memphis, one of Egypt's most important cities",
        "current_state": "Mostly ruins with some architectural elements remaining"
    },
    {
        "name": "Tomb of Perneb",
        "city": "Saqqara",
        "description": "Old Kingdom mastaba tomb now reconstructed at the Metropolitan Museum of Art in New York.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "Old Kingdom",
        "dynasty": "5th Dynasty",
        "current_location": "Metropolitan Museum of Art, New York",
        "significance": "One of the few complete Egyptian tomb chapels outside Egypt"
    },
    {
        "name": "Sanctuary of Thoth at Tuna el-Gebel",
        "city": "Minya",
        "description": "Necropolis with catacombs containing millions of mummified ibises and baboons sacred to Thoth, god of wisdom.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Late Period to Roman",
        "unique_feature": "Catacombs with millions of animal mummies dedicated to Thoth",
        "additional_attraction": "Contains the tomb of Petosiris with Greek-influenced Egyptian art"
    },
    {
        "name": "Hermopolis (El-Ashmunein)",
        "city": "Minya",
        "description": "Ancient city dedicated to Thoth with remains of temples and the largest known ancient Egyptian statues of baboons.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "Various periods",
        "theological_significance": "Site associated with the Ogdoad creation myth and the primeval mound",
        "notable_remains": "Massive baboon statues, Roman basilica reusing pharaonic blocks"
    },
    {
        "name": "Sacred Lake of Karnak",
        "city": "Luxor",
        "description": "Large man-made lake within Karnak Temple complex used for ritual purification and sacred boat ceremonies.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "significance": "Used for ritual cleansing and as part of the sacred boat processions",
        "unique_feature": "Rectangular artificial lake lined with stone and featuring a nilometer"
    },
    {
        "name": "Tomb of Ramose",
        "city": "Luxor",
        "description": "Nobleman's tomb showing the transition between traditional and Amarna artistic styles during the reign of Akhenaten.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "historical_figure": "Ramose was Vizier during transition from Amenhotep III to Akhenaten",
        "significance": "Shows artistic transition between traditional and revolutionary Amarna styles"
    },
    {
        "name": "Temple of Hathor at Serabit el-Khadim",
        "city": "Sinai Peninsula",
        "description": "Remote temple in the turquoise mining region of Sinai with the earliest known alphabetic inscriptions (Proto-Sinaitic).",
        "type": "Pharaonic",
        "popularity": 2,
        "period": "Middle Kingdom to New Kingdom",
        "linguistic_significance": "Site of earliest known alphabetic inscriptions, precursor to many modern alphabets",
        "archaeological_importance": "Shows Egyptian mining operations and worker settlement"
    },
    {
        "name": "Malkata Palace",
        "city": "Luxor",
        "description": "Remains of the vast palace complex of Amenhotep III, once the largest royal residence in Egypt.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "significance": "Shows the luxury and scale of royal palaces rarely preserved in Egypt",
        "features": "Included audience halls, royal apartments, artificial harbor and lake"
    },
    {
        "name": "Tombs of el-Kab",
        "city": "El-Kab",
        "description": "Rock-cut tombs with important historical inscriptions, including an eyewitness account of the expulsion of the Hyksos invaders.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "historical_significance": "Contains autobiography of Ahmose, son of Ibana, with firsthand account of war against the Hyksos"
    },
    {
        "name": "Meidum Pyramid",
        "city": "Meidum",
        "description": "Collapsed step pyramid showing transition in pyramid design, now resembling a tower on a mountain of debris.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Old Kingdom",
        "dynasty": "4th Dynasty",
        "attribution": "Started by Huni, completed by Sneferu",
        "significance": "Shows experimental stage in evolution of true pyramids",
        "current_state": "Partially collapsed, revealing internal structure rarely seen"
    },
    {
        "name": "The Labyrinth at Hawara",
        "city": "Faiyum",
        "description": "Remains of what ancient writers described as an enormous palace complex with thousands of rooms and multiple levels.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Middle Kingdom",
        "dynasty": "12th Dynasty",
        "pharaoh": "Amenemhat III",
        "historical_accounts": "Described by Herodotus as surpassing the pyramids in grandeur",
        "current_state": "Mostly ruins with foundation outlines remaining"
    },
    {
        "name": "Tomb of Khnumhotep II",
        "city": "Beni Hasan",
        "description": "Rock-cut tomb with famous scene depicting Middle Eastern traders (possibly early Semitic peoples) visiting Egypt.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Middle Kingdom",
        "dynasty": "12th Dynasty",
        "significance": "Contains earliest known depiction of people who may be Canaanites or early Semitic peoples",
        "historical_figure": "Khnumhotep II was a nomarch (provincial governor) under Amenemhat II"
    },
    {
        "name": "Valley of the Queens",
        "city": "Luxor",
        "description": "Burial site of queens and royal children of the New Kingdom, including the spectacular tomb of Nefertari.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "New Kingdom",
        "dynasty": "19th-20th Dynasties",
        "notable_tombs": ["QV66 (Nefertari)", "QV44 (Khaemweset)", "QV55 (Amunherkhepshef)"],
        "significance": "Contains tombs of queens and royal children, including Nefertari's tomb with the best preserved paintings"
    },
    {
        "name": "Pyramids of Giza Sound and Light Show",
        "city": "Giza",
        "description": "Nighttime spectacle that brings ancient history to life through dramatic narration, music, and illumination of the pyramids and Sphinx.",
        "type": "Pharaonic",
        "popularity": 8,
        "modern_feature": "Uses advanced lighting and projection technology to tell the story of ancient Egypt",
        "languages_available": ["English", "Arabic", "French", "Spanish", "German", "Italian", "Japanese"]
    },
    {
        "name": "Tomb of Merenptah",
        "city": "Luxor",
        "description": "Burial place of the 13th son and successor of Ramses II, featuring the famous 'Israel Stela'.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "historical_significance": "His victory stela contains the earliest known reference to Israel as a people"
    },
    {
        "name": "Tombs of the Workers at Deir el-Medina",
        "city": "Luxor",
        "description": "Beautifully decorated tombs of the skilled artisans who created the royal tombs in the Valley of the Kings.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom",
        "dynasty": "18th-20th Dynasties",
        "significance": "Shows how ordinary skilled workers were buried with elaborate decorations",
        "notable_tombs": ["Tomb of Sennedjem", "Tomb of Pashedu", "Tomb of Inherkhau"]
    },
    {
        "name": "The Unfinished Obelisk",
        "city": "Aswan",
        "description": "Enormous obelisk abandoned in the quarry when cracks appeared, providing insights into ancient stoneworking techniques.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "attribution": "Likely commissioned by Hatshepsut",
        "significance": "Reveals ancient quarrying and carving techniques for megalithic monuments",
        "dimensions": "Would have been 42 meters tall and weighed 1,200 tons if completed"
    },
    {
        "name": "Wadi Hammamat",
        "city": "Eastern Desert",
        "description": "Ancient quarry site with thousands of inscriptions spanning from prehistoric times to the Roman period.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "Multiple periods",
        "significance": "Contains over 4,000 rock inscriptions documenting quarrying expeditions",
        "historical_importance": "Major source of greywacke stone used for royal statuary and sarcophagi"
    },
    {
        "name": "Elephantine Island",
        "city": "Aswan",
        "description": "Island with ruins of the Temple of Khnum and a nilometer used to measure the Nile flood levels.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Multiple periods",
        "significance": "Was Egypt's southern frontier for much of its history",
        "unique_feature": "Home to the temples of Satet and Khnum and a rare surviving Nilometer"
    },
    {
        "name": "Abu Gorab Sun Temple",
        "city": "Abusir",
        "description": "Remains of a temple dedicated to Ra built by King Nyuserre with a large alabaster altar and symbolic obelisk.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Old Kingdom",
        "dynasty": "5th Dynasty",
        "significance": "One of only two surviving sun temples from the Old Kingdom",
        "unique_feature": "Featured a truncated obelisk (benben) standing on a platform"
    },
    {
        "name": "Temple of Hathor at Timna",
        "city": "Sinai Peninsula",
        "description": "Egyptian temple to Hathor in a copper mining region, later converted to a Midianite shrine.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "New Kingdom",
        "dynasty": "19th-20th Dynasties",
        "location_note": "Located in modern Israel, was part of Egypt's empire",
        "significance": "Shows Egyptian religious presence at distant mining operations"
    },
    {
        "name": "Temple of Isis at Behbeit el-Hagar",
        "city": "Nile Delta",
        "description": "Ruins of a major temple to Isis built entirely of granite blocks, now mostly collapsed.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "Late Period to Ptolemaic",
        "unique_feature": "Unusual use of granite for entire temple construction",
        "current_state": "Collapsed ruins with massive granite blocks scattered across the site"
    },
    {
        "name": "El-Kab Temple Enclosure",
        "city": "El-Kab",
        "description": "Massive mud-brick enclosure wall surrounding the ancient city with temples to local goddess Nekhbet.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Multiple periods",
        "significance": "Dedicated to Nekhbet, the vulture goddess and protector of Upper Egypt",
        "unique_feature": "Enormous mud-brick walls still standing to considerable height"
    },
    {
        "name": "Tombs of Qubbet el-Hawa",
        "city": "Aswan",
        "description": "Rock-cut tombs of provincial governors carved into the cliffs opposite Aswan, with biographical inscriptions.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Old Kingdom to Middle Kingdom",
        "significance": "Contains autobiographies of officials who led expeditions to Nubia and Punt",
        "notable_tomb": "Tomb of Harkhuf, who brought a 'dancing dwarf' from central Africa"
    },
    {
        "name": "Temple of Amenhotep III",
        "city": "Luxor",
        "description": "Remains of a massive temple with the Colossi of Memnon as its entrance guardians, mostly destroyed by annual Nile floods.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "New Kingdom", 
        "dynasty": "18th Dynasty",
        "current_state": "Mostly destroyed with ongoing excavation revealing foundations",
        "significance": "Was once the largest temple complex in Egypt, surpassing even Karnak"
    },
    {
        "name": "Temple of Debod",
        "city": "Madrid, Spain (originally from Aswan)",
        "description": "Complete temple saved from Lake Nasser and gifted to Spain, now reconstructed in a park in Madrid.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Ptolemaic to Roman",
        "relocation": "Gifted to Spain in 1968 as thanks for helping save Abu Simbel",
        "significance": "One of the few complete Egyptian temples outside Egypt"
    },
    {
        "name": "Tomb of Maya and Meryt",
        "city": "Saqqara",
        "description": "New Kingdom tomb of Tutankhamun's treasurer and his wife, with beautiful reliefs and statues.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "historical_figure": "Maya was treasurer and overseer of works for Tutankhamun",
        "discovery": "Rediscovered in 1986 by a joint Dutch-British team"
    },
    {
        "name": "Temple of Taweret at Gebelein",
        "city": "Gebelein",
        "description": "Temple dedicated to the hippopotamus goddess of childbirth and fertility.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "Multiple periods",
        "significance": "Rare temple dedicated primarily to the protective goddess of pregnant women"
    },
    {
        "name": "Kiosk of Qertassi",
        "city": "Lake Nasser (relocated)",
        "description": "Small but elegant Roman-period kiosk with delicate floral capitals, saved from the waters of Lake Nasser.",
        "type": "Pharaonic",
        "popularity": 4,
        "period": "Roman",
        "significance": "Example of blended Egyptian and Greco-Roman architectural styles",
        "relocation": "Moved to New Kalabsha site to save it from Lake Nasser's waters"
    },
    {
        "name": "Tombs of Asyut",
        "city": "Asyut",
        "description": "Rock-cut tombs of local governors with important texts about the turbulent First Intermediate Period.",
        "type": "Pharaonic",
        "popularity": 3,
        "period": "First Intermediate Period to Middle Kingdom",
        "significance": "Contains rare accounts of the chaotic period between the Old and Middle Kingdoms"
    },
    {
        "name": "Temple of Mandulis at Kalabsha",
        "city": "Lake Nasser (relocated)",
        "description": "Largest freestanding Nubian temple, saved and moved to higher ground before Lake Nasser formed.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Roman",
        "significance": "Shows Egyptian religious traditions continuing into Roman period",
        "relocation": "Entirely dismantled and moved in 1963 to save it from Lake Nasser's waters"
    },
    {
        "name": "Tombs of the Apis Bulls",
        "city": "Saqqara",
        "description": "Underground galleries containing massive stone sarcophagi for the mummified sacred bulls of Memphis.",
        "type": "Pharaonic",
        "popularity": 5,
        "period": "Late Period to Ptolemaic",
        "significance": "Shows importance of animal cults in later Egyptian religion",
        "enigmatic_feature": "Each sarcophagus weighs over 70 tons and was moved through narrow tunnels"
    },
    {
        "name": "Opet Festival Reliefs at Luxor Temple",
        "city": "Luxor",
        "description": "Detailed reliefs depicting the important annual Opet Festival when the statue of Amun traveled from Karnak to Luxor Temple.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "significance": "Provides detailed visual record of one of Egypt's most important religious festivals"
    }
]
attractions_df = pd.DataFrame(ATTRACTIONS_DATA)

RECOMMENDATION_SYSTEM_MODEL_ST = None
ATTRACTION_EMBEDDINGS = None
RECOMMENDATION_SYSTEM_READY = False
print("Loading sentence transformer model for recommendations...")
try:
    RECOMMENDATION_SYSTEM_MODEL_ST = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("Recommendation sentence transformer model loaded successfully!")
    print("Generating embeddings for attractions...")
    ATTRACTION_EMBEDDINGS = RECOMMENDATION_SYSTEM_MODEL_ST.encode(attractions_df['description'].tolist())
    print(f"Generated {len(ATTRACTION_EMBEDDINGS)} embeddings with dimension {ATTRACTION_EMBEDDINGS.shape[1]}")
    RECOMMENDATION_SYSTEM_READY = True
except Exception as e:
    print(f"Error loading sentence transformer model or generating embeddings: {e}")
    print("Recommendation system will not be fully functional. Please ensure 'sentence-transformers', 'pandas', and 'scikit-learn' are installed.")

# --- Original Image Classification Model Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "last_model_bgd.keras")
image_classification_model = None
if os.path.exists(MODEL_PATH):
    try:
        image_classification_model = load_model(MODEL_PATH)
        print(f"Image classification model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading Keras model from {MODEL_PATH}: {e}")
        image_classification_model = None
else:
    print(f"Warning: Image classification model file not found at {MODEL_PATH}. Classification will be mocked.")

# --- Original Image Classification Functions (Unchanged) ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_bytes):
    if image_classification_model is None:
        print("Image classification model not loaded, using mock classification.")
        # Ensure class_names is not empty before accessing
        cn = class_names[0] if class_names and len(class_names) > 0 else "Mocked Artifact"
        return cn, f"This is a mocked English description for {cn} as the model is not available."
    
    if not image_bytes:
        return "Error: No image data", ""

    try:
        preprocessed_image = preprocess_image(image_bytes)
        predictions = image_classification_model.predict(preprocessed_image)
        class_idx = int(np.argmax(predictions[0]))
        
        if 0 <= class_idx < len(class_names):
            predicted_class_name = class_names[class_idx]
        else:
            predicted_class_name = "Unknown Artifact"
            print(f"Warning: Predicted class index {class_idx} is out of bounds for class_names.")

        description = f"This is a magnificent {predicted_class_name}, a true masterpiece of ancient Egyptian art, reflecting the rich history and culture of the civilization."
        return predicted_class_name, description
    except Exception as e:
        print(f"Error during image classification: {e}")
        return "Error during classification", str(e)

# --- New Recommendation Function (Replaces get_recommendations_mock) ---
def generate_text_recommendations(current_location, interests, liked_places=None, top_n=3):
    if not RECOMMENDATION_SYSTEM_READY or RECOMMENDATION_SYSTEM_MODEL_ST is None or ATTRACTION_EMBEDDINGS is None:
        return "Recommendation system is currently unavailable. Please check logs for model loading errors."
    
    if isinstance(interests, str):
        interests = [interest.strip() for interest in interests.split(',') if interest.strip()]
    if not interests: # Default interests if none provided
        interests = ["egyptian history", "culture"]
        
    interests = [interest.lower() for interest in interests]
    
    recommendations = attractions_df.copy()
    
    if current_location and current_location.lower() != 'all' and current_location.lower() != 'any':
        recommendations['location_score'] = (recommendations['city'].str.lower() == current_location.lower()).astype(int)
    else:
        recommendations['location_score'] = 1
    
    interests_text = " ".join(interests)
    interests_embedding = RECOMMENDATION_SYSTEM_MODEL_ST.encode([interests_text])[0]
    
    similarity_scores = cosine_similarity([interests_embedding], ATTRACTION_EMBEDDINGS)[0]
    recommendations['interest_score'] = similarity_scores
    
    if liked_places and len(liked_places) > 0:
        liked_indices = []
        for place in liked_places:
            indices = recommendations.index[recommendations['name'].str.lower() == place.lower()].tolist()
            liked_indices.extend(indices)
        
        if liked_indices:
            liked_embeddings_val = ATTRACTION_EMBEDDINGS[liked_indices]
            history_scores = cosine_similarity(ATTRACTION_EMBEDDINGS, liked_embeddings_val).mean(axis=1)
            recommendations['history_score'] = history_scores
        else:
            recommendations['history_score'] = 0
    else:
        recommendations['history_score'] = 0
    
    recommendations['final_score'] = (
        0.2 * recommendations['location_score'] +
        0.5 * recommendations['interest_score'] +
        0.2 * recommendations['history_score'] +
        0.1 * (recommendations['popularity'] / 10)
    )
    
    top_recommendations_df = recommendations.sort_values('final_score', ascending=False).head(top_n)
    
    if top_recommendations_df.empty:
        return "No specific recommendations found based on your preferences. Try broadening your search or checking your spelling!"

    results_text = "Top Recommended Egyptian Attractions for you:\n\n"
    for i, (idx, row) in enumerate(top_recommendations_df.iterrows(), 1):
        results_text += f"{i}. {row['name']} ({row['city']}) - {row['type']}\n"
        # Ensure final_score is present and calculate match_score
        match_score = round(row["final_score"] * 100, 1) if "final_score" in row and isinstance(row["final_score"], (int, float)) else "N/A"
        results_text += f"   Match Score: {match_score}%\n"
        results_text += f"   Description: {row['description']}\n\n"
    
    return results_text.strip()

# --- Original Routes (Unchanged, except /get_recommendations) ---
@app.route("/")
def index():
    return render_template("horos1.html") # Original index page

@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

@app.route("/page2_image_result")
def result_page():
    return render_template("page2_image_result.html")

@app.route("/page3_recommendation_result")
def recommendation_display_page():
    return render_template("page3_recommendation_result.html")

@app.route("/upload_image", methods=["POST"])
def upload_image_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected for uploading"}), 400
    try:
        image_bytes = file.read()
        class_name, description = classify_image(image_bytes)
        if "Error" in class_name:
             return jsonify({"error": description or class_name}), 500
        return jsonify({"class_name": class_name, "description": description})
    except Exception as e:
        print(f"Error in /upload_image route: {e}")
        return jsonify({"error": "An unexpected error occurred during image processing."}), 500

# --- Modified /get_recommendations Route ---
@app.route("/get_recommendations", methods=["POST"])
def get_recommendations_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided for recommendations"}), 400
        
    location = data.get("location")
    interests = data.get("interests") # Expected to be a string like "history, culture"
    liked_places_input = data.get("liked_places") # Expected to be a string like "Pyramids, Museum"
    # Duration is no longer used by the new recommendation logic directly for structuring itinerary
    # duration = data.get("duration")

    liked_places_list = []
    if isinstance(liked_places_input, str) and liked_places_input.strip():
        liked_places_list = [p.strip() for p in liked_places_input.split(',') if p.strip()]
    elif isinstance(liked_places_input, list):
        liked_places_list = [str(p).strip() for p in liked_places_input if str(p).strip()] # Ensure strings

    current_location_param = location if location else "All" # Default to 'All' if not provided
    interests_param = interests if interests else "history, culture" # Default interests

    try:
        recommendations_text_output = generate_text_recommendations(
            current_location_param, 
            interests_param, 
            liked_places_list,
            top_n=3 # Number of recommendations to return
        )
        return jsonify({"recommendations": recommendations_text_output})
    except Exception as e:
        print(f"Error in /get_recommendations route: {e}")
        fallback_message = "We encountered an issue generating recommendations. Please try again later."
        if not RECOMMENDATION_SYSTEM_READY:
            fallback_message = "Recommendation system is currently initializing or unavailable. Please try again shortly."
        # Return the error and a fallback message in the expected format
        return jsonify({"error": str(e), "recommendations": fallback_message}), 500

# --- Original Chat Route (Unchanged) ---
@app.route("/chat_with_horus", methods=["POST"])
def chat_with_horus_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided for chat"}),400
        
    user_message = data.get("user_message")
    artifact_name = data.get("artifact_name")
    artifact_description = data.get("artifact_description")

    if not user_message or not artifact_name or not artifact_description:
        return jsonify({"error": "Missing required fields for chat (user_message, artifact_name, artifact_description)"}), 400

    try:
        # Check if generate_chat_response is the placeholder or the actual one
        if 'generate_chat_response' in globals() and callable(generate_chat_response):
            bot_response = generate_chat_response(user_message, artifact_name, artifact_description)
        else: # Should not happen if placeholder is defined correctly
            bot_response = "Chat functionality is currently unavailable."
        return jsonify({"bot_response": bot_response})
    except Exception as e:
        print(f"Error in /chat_with_horus route: {e}")
        return jsonify({"error": "An unexpected error occurred while generating chat response."}), 500

# --- Combined __main__ Block ---
if __name__ == "__main__":
    if image_classification_model is None:
        print("IMPORTANT: The Keras model file 'last_model_bgd.keras' was not found or failed to load.")
        print("The application will run with MOCKED image classification results.")
    else:
        print("Image classification Keras model loaded.")
        
    if not RECOMMENDATION_SYSTEM_READY:
        print("IMPORTANT: The recommendation system may not be fully functional due to model loading issues.")
        print("Please check messages above for errors related to 'sentence-transformers'.")
        print("You might need to install necessary packages: pip install sentence-transformers pandas scikit-learn tensorflow Pillow")
    else:
        print("Recommendation system ready.")
        
    app.run(debug=True, port=5000)

