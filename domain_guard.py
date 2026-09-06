import os
import re
import unicodedata

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ==================================================
# NORMALISATION
# ==================================================

def normalize_text(text: str) -> str:
    """
    Normalise le texte pour faciliter la détection
    des termes du domaine.
    """

    text = text.lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = re.sub(
        r"[^a-z0-9\s\-']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ==================================================
# VOCABULAIRE FORT DU DOMAINE
# ==================================================
#
# Ces termes permettent d'accepter immédiatement
# les questions clairement liées au vin.
#
# Les termes trop ambigus comme "bordeaux",
# "champagne", "cave", etc. sont volontairement
# séparés plus bas.
# ==================================================

STRONG_WINE_TERMS = {

    # --------------------------------------------------
    # VIN / VIGNE
    # --------------------------------------------------

    "vin",
    "vins",
    "wine",
    "wines",
    "vigne",
    "vignes",
    "vine",
    "vines",
    "vignoble",
    "vignobles",
    "vineyard",
    "vineyards",
    "viticulture",
    "viticole",
    "viticoles",
    "viticulteur",
    "viticulteurs",
    "viticultrice",
    "vigneron",
    "vignerons",
    "vigneronne",
    "vigneronnes",
    "oenologie",
    "oenologique",
    "oenologiques",
    "oenologue",
    "oenologues",
    "winemaking",
    "winemaker",
    "winemakers",

    # --------------------------------------------------
    # RAISIN / BAIE / PLANTE
    # --------------------------------------------------

    "raisin",
    "raisins",
    "grape",
    "grapes",
    "grappe",
    "grappes",
    "baie",
    "baies",
    "cep",
    "ceps",
    "sarment",
    "sarments",
    "bourgeon",
    "bourgeons",
    "bourgeonnement",
    "debourrement",
    "floraison",
    "veraison",
    "maturite",
    "maturite phenolique",
    "maturite technologique",

    # --------------------------------------------------
    # CEPAGES
    # --------------------------------------------------

    "cepage",
    "cepages",
    "variete de raisin",
    "varietes de raisin",
    "grape variety",
    "grape varieties",

    "merlot",
    "cabernet sauvignon",
    "cabernet franc",
    "malbec",
    "petit verdot",
    "carmenere",
    "pinot noir",
    "pinot gris",
    "pinot blanc",
    "chardonnay",
    "sauvignon blanc",
    "semillon",
    "muscadelle",
    "riesling",
    "gewurztraminer",
    "chenin",
    "chenin blanc",
    "syrah",
    "shiraz",
    "grenache",
    "mourvedre",
    "carignan",
    "cinsault",
    "gamay",
    "nebbiolo",
    "sangiovese",
    "tempranillo",
    "touriga nacional",
    "viognier",
    "marsanne",
    "roussanne",
    "melon de bourgogne",
    "aligote",
    "tannat",

    # --------------------------------------------------
    # VENDANGE
    # --------------------------------------------------

    "vendange",
    "vendanges",
    "vendanger",
    "vendangeur",
    "recolte du raisin",
    "harvest",
    "grape harvest",
    "vendange tardive",
    "vendanges tardives",
    "tri du raisin",
    "table de tri",
    "egrappage",
    "erafflage",
    "foulage",
    "pressoir",
    "pressurage",

    # --------------------------------------------------
    # VINIFICATION
    # --------------------------------------------------

    "vinification",
    "fermentation alcoolique",
    "fermentation malolactique",
    "fermentation malo lactique",
    "malolactique",
    "malo",
    "levure",
    "levures",
    "yeast",
    "yeasts",
    "saccharomyces",
    "saccharomyces cerevisiae",
    "bacterie lactique",
    "bacteries lactiques",
    "moût",
    "mout",
    "must",
    "jus de raisin",
    "macération",
    "maceration",
    "maceration carbonique",
    "maceration pelliculaire",
    "remontage",
    "pigeage",
    "delestage",
    "cuvaison",
    "encuvage",
    "decuvage",
    "ecoulage",
    "pressurage",
    "clarification",
    "collage",
    "filtration",
    "centrifugation",
    "stabilisation",
    "stabilisation tartrique",
    "stabilisation proteique",
    "soutirage",
    "assemblage",
    "coupage",
    "elevage",
    "batonnage",
    "lies",
    "lies fines",
    "autolyse",
    "prise de mousse",
    "tirage",
    "remuage",
    "degorgement",
    "dosage",
    "liqueur de tirage",
    "liqueur d expedition",
    "liqueur d'expedition",

    # --------------------------------------------------
    # CUVES / FÛTS / BOIS
    # --------------------------------------------------

    "cuve",
    "cuves",
    "cuverie",
    "cuve inox",
    "cuve beton",
    "amphore",
    "jarre",
    "barrique",
    "barriques",
    "fut",
    "futs",
    "tonneau",
    "tonneaux",
    "tonnellerie",
    "tonnelier",
    "chene",
    "chene francais",
    "chene americain",
    "chauffe du bois",
    "chauffe de barrique",
    "grain du bois",
    "micro oxygenation",
    "microoxygenation",

    # --------------------------------------------------
    # CHIMIE DU VIN
    # --------------------------------------------------

    "tanin",
    "tanins",
    "tannin",
    "tannins",
    "polyphenol",
    "polyphenols",
    "anthocyane",
    "anthocyanes",
    "flavonoide",
    "flavonoides",
    "proanthocyanidine",
    "proanthocyanidines",
    "ellagitanin",
    "ellagitanins",
    "gallotanin",
    "gallotanins",
    "acide tartrique",
    "acide malique",
    "acide lactique",
    "acide acetique",
    "acidite volatile",
    "acidite totale",
    "acidite titrable",
    "ph du vin",
    "ethanol",
    "alcool du vin",
    "degre alcoolique",
    "titre alcoometrique",
    "sucre residuel",
    "sucres residuels",
    "glucose",
    "fructose",
    "glycerol",
    "sulfite",
    "sulfites",
    "dioxyde de soufre",
    "so2",
    "oxygene dissous",
    "oxidation",
    "reduction",
    "redox",
    "acetaldehyde",
    "acetaldehyde",
    "ester",
    "esters",
    "thiol",
    "thiols",
    "terpene",
    "terpenes",
    "methoxypyrazine",
    "methoxypyrazines",

    # --------------------------------------------------
    # AROMES / SENSORIEL
    # --------------------------------------------------

    "arome du vin",
    "aromes du vin",
    "arome varietal",
    "aromes varietaux",
    "arome fermentaire",
    "aromes fermentaires",
    "bouquet",
    "nez du vin",
    "palais",
    "bouche du vin",
    "degustation",
    "deguster",
    "degustateur",
    "degustatrice",
    "analyse sensorielle",
    "sensoriel",
    "sensorielle",
    "organoleptique",
    "astringence",
    "astringent",
    "amertume",
    "rondeur",
    "corps du vin",
    "structure du vin",
    "longueur en bouche",
    "finale du vin",
    "retro olfaction",
    "retro-olfaction",
    "robe du vin",
    "couleur du vin",
    "limpidite",
    "brillance",
    "larmes du vin",
    "jambes du vin",

    # --------------------------------------------------
    # DÉFAUTS ET ALTÉRATIONS
    # --------------------------------------------------

    "defaut du vin",
    "defauts du vin",
    "vin bouche",
    "gout de bouchon",
    "bouchonne",
    "bouchon tca",
    "tca",
    "brettanomyces",
    "brett",
    "volatile acidity",
    "acidite volatile",
    "oxydation du vin",
    "vin oxyde",
    "reduction du vin",
    "mercaptan",
    "mercaptans",
    "sulfure d hydrogene",
    "sulfure d'hydrogene",
    "h2s",
    "ethyl phenol",
    "ethylphenol",

    # --------------------------------------------------
    # TERROIR / SOL / CLIMAT
    # --------------------------------------------------

    "terroir",
    "terroirs",
    "terroir viticole",
    "sol viticole",
    "sols viticoles",
    "calcaire viticole",
    "argile viticole",
    "graves viticoles",
    "schiste viticole",
    "granite viticole",
    "exposition de la vigne",
    "parcelle viticole",
    "parcellaire",
    "microclimat viticole",
    "mesoclimat",
    "climat viticole",
    "changement climatique viticulture",
    "rechauffement climatique vin",
    "stress hydrique vigne",
    "stress hydrique de la vigne",
    "hydrologie viticole",

    # --------------------------------------------------
    # CULTURE DE LA VIGNE
    # --------------------------------------------------

    "taille de la vigne",
    "taille guyot",
    "guyot",
    "cordon de royat",
    "palissage",
    "effeuillage",
    "epamprement",
    "eclaircissage",
    "vendange en vert",
    "enherbement",
    "couvert vegetal vigne",
    "viticulture biologique",
    "vin biologique",
    "vin bio",
    "biodynamie",
    "biodynamique",
    "agroecologie viticole",
    "agroforesterie viticole",
    "viticulture regenerative",
    "irrigation de la vigne",

    # --------------------------------------------------
    # MALADIES DE LA VIGNE
    # --------------------------------------------------

    "phylloxera",
    "mildiou",
    "oidium",
    "botrytis",
    "pourriture noble",
    "pourriture grise",
    "black rot",
    "esca",
    "flavescence doree",
    "court noue",
    "maladie de la vigne",
    "maladies de la vigne",

    # --------------------------------------------------
    # VIEILLISSEMENT / GARDE
    # --------------------------------------------------

    "vieillissement du vin",
    "vieillissement des vins",
    "garde du vin",
    "potentiel de garde",
    "vin de garde",
    "vins de garde",
    "maturation du vin",
    "evolution en bouteille",
    "vieillissement en bouteille",
    "vieillissement en barrique",
    "elevage sous bois",

    # --------------------------------------------------
    # BOUTEILLE / BOUCHAGE
    # --------------------------------------------------

    "bouteille de vin",
    "bouteilles de vin",
    "mise en bouteille",
    "embouteillage",
    "bouchon de liege",
    "bouchon liege",
    "liege du vin",
    "capsule a vis",
    "screwcap",
    "bouchage du vin",
    "magnum de vin",
    "jeroboam vin",
    "methuselah vin",
    "imperiale vin",

    # --------------------------------------------------
    # CONSERVATION / SERVICE
    # --------------------------------------------------

    "conservation du vin",
    "conserver le vin",
    "temperature de service",
    "temperature du vin",
    "service du vin",
    "servir le vin",
    "carafer",
    "carafage",
    "decantation du vin",
    "decanter le vin",
    "aeration du vin",
    "verre a vin",
    "verres a vin",
    "sommelier",
    "sommeliere",
    "sommellerie",
    "accord mets vins",
    "accord mets et vins",
    "food pairing wine",

    # --------------------------------------------------
    # TYPES DE VIN
    # --------------------------------------------------

    "vin rouge",
    "vin blanc",
    "vin rose",
    "vin effervescent",
    "vin tranquille",
    "vin doux",
    "vin liquoreux",
    "vin moelleux",
    "vin sec",
    "vin orange",
    "vin nature",
    "vin naturel",
    "vin petillant",
    "vin mousseux",
    "vin fortifie",
    "vin de voile",
    "vin jaune",
    "vin de glace",
    "icewine",
    "pet nat",
    "pet-nat",
    "methode traditionnelle",
    "methode ancestrale",

    # --------------------------------------------------
    # CLASSIFICATION / ORIGINE
    # --------------------------------------------------

    "appellation viticole",
    "appellations viticoles",
    "appellation d origine",
    "appellation d'origine",
    "aoc vin",
    "aop vin",
    "igp vin",
    "indication geographique vin",
    "cru viticole",
    "grand cru",
    "premier cru",
    "cru classe",
    "classement des vins",
    "millesime",
    "millesimes",
    "vintage wine",

    # --------------------------------------------------
    # REGIONS VITICOLES
    # --------------------------------------------------

    "medoc",
    "saint emilion",
    "saint-emilion",
    "pomerol",
    "pauillac",
    "margaux vin",
    "saint julien vin",
    "saint-julien vin",
    "saint estephe vin",
    "saint-estephe vin",
    "pessac leognan",
    "pessac-leognan",
    "sauternes vin",
    "barsac vin",
    "entre deux mers vin",
    "entre-deux-mers",
    "bourgogne vin",
    "beaujolais vin",
    "alsace vin",
    "jura vin",
    "loire vin",
    "rhone vin",
    "vallee du rhone vin",
    "languedoc vin",
    "roussillon vin",
    "provence vin",
    "cognac vin",
    "rioja vin",
    "ribera del duero",
    "toscane vin",
    "piemont vin",
    "napa valley wine",
    "sonoma wine",
    "mendoza wine",
    "mosel wine",
    "douro wine",
    "port wine",
    "porto wine",

    # --------------------------------------------------
    # SCIENCE / MICROBIOLOGIE
    # --------------------------------------------------

    "microbiologie du vin",
    "microbiologie de la vigne",
    "microorganisme du vin",
    "microorganismes du vin",
    "microbiote du raisin",
    "microbiote viticole",
    "metabolisme des levures",
    "oenococcus oeni",
    "malolactic fermentation",

    # --------------------------------------------------
    # HISTOIRE ET CULTURE DU VIN
    # --------------------------------------------------

    "histoire du vin",
    "histoire de la vigne",
    "culture du vin",
    "culture viticole",
    "patrimoine viticole",
    "civilisation du vin",
    "route des vins",
    "commerce du vin",
    "histoire de l oenologie",
    "histoire de l'oenologie",
    "dionysos vin",
    "bacchus vin",
    "amphore vin",
    "vin romain",
    "vin antiquite",

    # --------------------------------------------------
    # ÉCONOMIE / MÉTIERS / RÉGLEMENTATION
    # --------------------------------------------------

    "marche du vin",
    "economie du vin",
    "filiere viticole",
    "filiere vin",
    "negoce du vin",
    "negociant en vin",
    "courtier en vin",
    "caviste",
    "maitre de chai",
    "chai viticole",
    "oenotourisme",
    "tourisme viticole",
    "wine tourism",
    "reglementation du vin",
    "etiquetage du vin",
    "etiquette de vin",
    "label vin",
    "certification viticole",

    # --------------------------------------------------
    # ENVIRONNEMENT
    # --------------------------------------------------

    "empreinte carbone du vin",
    "empreinte environnementale du vin",
    "biodiversite viticole",
    "biodiversite dans les vignes",
    "pesticide vigne",
    "pesticides vigne",
    "intrant oenologique",
    "intrants oenologiques",
    "intrant viticole",
    "intrants viticoles",
    "durabilite viticole",
    "viticulture durable",

    # --------------------------------------------------
    # ANALYSE DU VIN
    # --------------------------------------------------

    "analyse du vin",
    "analyse oenologique",
    "laboratoire oenologique",
    "densite du mout",
    "densite du vin",
    "refractometre vigne",
    "refractometre raisin",
    "brix raisin",
    "degre brix",
    "spectrometrie vin",
    "chromatographie vin",
    "analyse chimique vin"
}


# ==================================================
# TERMES AMBIGUS
# ==================================================
#
# Ils peuvent être liés au vin mais également à
# autre chose. Ils nécessitent donc une validation
# sémantique.
# ==================================================

AMBIGUOUS_WINE_TERMS = {

    "bordeaux",
    "champagne",
    "cognac",
    "porto",
    "port",
    "madeira",
    "madere",
    "jerez",
    "sherry",
    "marsala",
    "cave",
    "caves",
    "chai",
    "chateau",
    "domaine",
    "propriete",
    "cru",
    "margaux",
    "sauternes",
    "barsac",
    "saumur",
    "chinon",
    "sancerre",
    "chablis",
    "beaujolais",
    "bourgogne",
    "provence",
    "alsace",
    "jura",
    "rhone",
    "loire"
}


# ==================================================
# DÉTECTION LEXICALE
# ==================================================

def contains_strong_wine_term(
    question: str
) -> bool:

    q = normalize_text(
        question
    )

    padded_question = (
        f" {q} "
    )

    for term in STRONG_WINE_TERMS:

        normalized_term = normalize_text(
            term
        )

        # Mot ou expression complète.
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalized_term)
            + r"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            padded_question
        ):
            return True

    return False


def contains_ambiguous_wine_term(
    question: str
) -> bool:

    q = normalize_text(
        question
    )

    padded_question = (
        f" {q} "
    )

    for term in AMBIGUOUS_WINE_TERMS:

        normalized_term = normalize_text(
            term
        )

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalized_term)
            + r"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            padded_question
        ):
            return True

    return False


# ==================================================
# CLASSIFICATION SÉMANTIQUE
# ==================================================

def semantic_wine_domain_check(
    question: str
) -> bool:
    """
    Vérifie sémantiquement si la question appartient
    au domaine autorisé.

    Politique volontairement restrictive :
    en cas de doute, répondre NON.
    """

    prompt = (
        "Tu es un garde-fou de domaine pour un assistant scientifique "
        "de la Cité du Vin.\n\n"

        "Tu dois déterminer si la question utilisateur appartient "
        "DIRECTEMENT au domaine du vin.\n\n"

        "DOMAINE AUTORISÉ :\n"
        "- vin et vins ;\n"
        "- vigne, raisin et viticulture ;\n"
        "- œnologie et vinification ;\n"
        "- fermentation et microbiologie liées au vin ;\n"
        "- cépages ;\n"
        "- terroirs viticoles ;\n"
        "- sols et climat lorsqu'ils concernent la vigne ou le vin ;\n"
        "- chimie et biochimie du vin ;\n"
        "- tanins, polyphénols, acidité, alcool et composés du vin ;\n"
        "- arômes et analyse sensorielle du vin ;\n"
        "- vieillissement, élevage, barriques et conservation ;\n"
        "- service, dégustation, sommellerie et accords mets-vins ;\n"
        "- appellations, régions et classifications viticoles ;\n"
        "- histoire, culture et patrimoine du vin ;\n"
        "- maladies de la vigne ;\n"
        "- changement climatique appliqué à la viticulture ;\n"
        "- environnement et durabilité appliqués à la filière vin ;\n"
        "- métiers, économie et réglementation de la filière vin ;\n"
        "- œnotourisme et patrimoine vitivinicole.\n\n"

        "DOMAINE NON AUTORISÉ :\n"
        "- automobile et mécanique automobile ;\n"
        "- informatique générale ;\n"
        "- politique ;\n"
        "- médecine sans relation directe avec le vin ;\n"
        "- sport ;\n"
        "- finance générale ;\n"
        "- voyages sans rapport avec le vin ;\n"
        "- cuisine sans rapport avec le vin ;\n"
        "- histoire générale sans rapport avec le vin ;\n"
        "- toute autre question sans relation directe avec "
        "le vin ou la vigne.\n\n"

        "IMPORTANT :\n"
        "Une simple mention d'un lieu viticole ne suffit pas.\n"
        "Par exemple :\n"
        "- 'Quel moteur équipe la Porsche 911 ?' = NON\n"
        "- 'Quelle est la population de Bordeaux ?' = NON\n"
        "- 'Quels sont les vins de Bordeaux ?' = OUI\n"
        "- 'Pourquoi les sols calcaires influencent-ils la vigne ?' = OUI\n"
        "- 'Comment fonctionne une pompe à injection diesel ?' = NON\n"
        "- 'Comment fonctionne une pompe utilisée pendant la vinification ?' = OUI\n\n"

        "En cas de doute, réponds NON.\n\n"

        "Réponds avec exactement un seul mot : "
        "OUI ou NON.\n\n"

        f"QUESTION :\n{question}"
    )

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        result = (
            response.output_text
            .strip()
            .upper()
        )

        return result == "OUI"

    except Exception:

        # Fail closed :
        # aucune question incertaine ne contourne
        # le garde-fou en cas de panne.
        return False


# ==================================================
# GARDE-FOU PRINCIPAL
# ==================================================

def is_wine_domain(
    question: str
) -> bool:
    """
    Retourne True uniquement si la question peut
    entrer dans le pipeline scientifique.
    """

    if not question:
        return False

    question = question.strip()

    if len(question) < 2:
        return False


    # --------------------------------------------------
    # Cas clairement œnologiques
    # --------------------------------------------------

    if contains_strong_wine_term(
        question
    ):
        return True


    # --------------------------------------------------
    # Cas ambigus ou sans mot-clé explicite
    #
    # On utilise le classifieur sémantique.
    # Cela permet par exemple de comprendre :
    #
    # "Pourquoi un sol calcaire change-t-il la
    # maturité des baies ?"
    #
    # même si le vocabulaire exact n'est pas dans
    # l'allowlist.
    # --------------------------------------------------

    return semantic_wine_domain_check(
        question
    )


# ==================================================
# MESSAGE DE REFUS
# ==================================================

def get_out_of_domain_message() -> str:

    return (
        "Je suis un agent scientifique spécialisé dans le vin, "
        "la vigne, la viticulture et l'œnologie. "
        "Je peux répondre aux questions portant notamment sur "
        "les cépages, les terroirs, la vinification, la fermentation, "
        "la dégustation, le vieillissement, les régions viticoles "
        "et la culture du vin."
    )