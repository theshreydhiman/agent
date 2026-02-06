"""Hashtag generation and strategy engine."""

import random

import structlog

from content.llm_client import LLMClient

logger = structlog.get_logger()

# Predefined hashtag pools by niche — works without LLM
NICHE_HASHTAGS: dict[str, dict[str, list[str]]] = {
    "lifestyle": {
        "high_volume": [
            "lifestyle", "dailylife", "lifestyleblogger", "instagood",
            "photooftheday", "beautiful", "happy", "love", "instadaily",
        ],
        "medium_volume": [
            "lifestyleinspo", "dailyinspiration", "livingmybestlife",
            "goodvibesonly", "positivevibes", "mindfulness", "selfcare",
            "wellbeing", "slowliving", "simpleliving",
        ],
        "niche": [
            "aestheticlife", "intentionalliving", "lifestyledesign",
            "everydaymoments", "consciousliving", "lifeunfiltered",
            "dailyroutine", "morningroutine", "lifestyletips", "livefully",
        ],
    },
    "fashion": {
        "high_volume": [
            "fashion", "style", "ootd", "fashionista", "streetstyle",
            "instafashion", "fashionblogger", "outfitoftheday", "stylish",
        ],
        "medium_volume": [
            "fashioninspo", "styleinspo", "fashiondaily", "whatiwore",
            "outfitideas", "fashionstyle", "lookoftheday", "styleguide",
            "fashiontrends", "dailyfashion",
        ],
        "niche": [
            "minimalfashion", "capsulewardrobe", "sustainablefashion",
            "slowfashion", "consciouscloset", "outfitinspiration",
            "styletips", "fashionforward", "wardrobeessentials", "chicstyle",
        ],
    },
    "fitness": {
        "high_volume": [
            "fitness", "gym", "workout", "fitnessmotivation", "fit",
            "fitspo", "training", "health", "exercise",
        ],
        "medium_volume": [
            "fitlife", "gymlife", "workoutmotivation", "fitfam",
            "healthylifestyle", "strengthtraining", "fitnessjourney",
            "activelife", "fitnessgirl", "workoutroutine",
        ],
        "niche": [
            "homeworkout", "functionalfitness", "mindfulmovement",
            "fitnesstips", "trainhard", "dailyworkout", "fitnessgoals",
            "strengthandconditioning", "bodyweighttraining", "wellnessjourney",
        ],
    },
    "travel": {
        "high_volume": [
            "travel", "wanderlust", "travelgram", "instatravel",
            "travelphotography", "explore", "adventure", "traveling",
        ],
        "medium_volume": [
            "travelinspo", "travelblogger", "traveltheworld", "passport",
            "traveler", "globetrotter", "traveladdict", "roamtheplanet",
            "beautifuldestinations", "travelholic",
        ],
        "niche": [
            "solotravel", "sustainabletravel", "slowtravel", "offthebeatenpath",
            "traveltips", "hiddenplaces", "traveldiaries", "budgettravel",
            "digitalnomad", "conscioustravel",
        ],
    },
    "beauty": {
        "high_volume": [
            "beauty", "makeup", "skincare", "beautyblogger", "cosmetics",
            "glam", "instamakeup", "beautytips", "mua",
        ],
        "medium_volume": [
            "beautyinspo", "makeuplover", "skincareroutine", "beautyroutine",
            "makeuptutorial", "beautycommunity", "skincareaddict",
            "naturalbeauty", "makeuplooks", "beautyproducts",
        ],
        "niche": [
            "cleanbeauty", "veganbeauty", "glowingskin", "minimalmakeup",
            "skincaretips", "beautyhacks", "everydaymakeup", "dewyglam",
            "skincareobsessed", "effortlessbeauty",
        ],
    },
    "food": {
        "high_volume": [
            "food", "foodie", "instafood", "foodporn", "yummy",
            "delicious", "foodphotography", "foodstagram", "cooking",
        ],
        "medium_volume": [
            "foodblogger", "homecooking", "foodlover", "healthyfood",
            "foodiesofinstagram", "easyrecipes", "comfortfood",
            "plantbased", "foodpics", "eatclean",
        ],
        "niche": [
            "foodstyling", "homechef", "seasonalcooking", "wholefood",
            "mealprep", "intuitiveeating", "farmtotable", "cookingtips",
            "foodfromhome", "simplerrecipes",
        ],
    },
    "tech": {
        "high_volume": [
            "tech", "technology", "innovation", "ai", "digital",
            "gadgets", "coding", "startup", "programming",
        ],
        "medium_volume": [
            "techlife", "futuretech", "techcommunity", "digitalnomad",
            "techreview", "artificialintelligence", "machinelearning",
            "softwaredeveloper", "techworld", "techtrends",
        ],
        "niche": [
            "aicreator", "generativeai", "techethics", "openai",
            "productivitytech", "techinnovation", "buildinginfluencer",
            "aiart", "techmindset", "futureready",
        ],
    },
}


class HashtagEngine:
    """Generates and manages hashtag strategy."""

    def __init__(self):
        self._llm = LLMClient()

    async def generate_hashtags(
        self, theme: str, niche: str, content_type: str, count: int = 25
    ) -> list[str]:
        """Generate hashtags using LLM with fallback to predefined pools."""
        prompt = (
            f"Generate {count} Instagram hashtags for a {content_type} post.\n"
            f"Niche: {niche}\nTheme: {theme}\n"
            f"Return ONLY a JSON array of hashtag strings without the # symbol.\n"
            f"Mix high-volume, medium, and niche hashtags."
        )

        try:
            response = await self._llm.generate(prompt, temperature=0.8)
            import json
            tags = json.loads(response)
            if isinstance(tags, list) and len(tags) >= 5:
                return [t.lstrip("#").strip() for t in tags[:count]]
        except Exception:
            logger.warning("llm_hashtag_generation_failed", theme=theme)

        return self.build_hashtag_set(niche, theme, count)

    def build_hashtag_set(
        self, niche: str, theme: str = "", count: int = 25
    ) -> list[str]:
        """Build optimized hashtag mix from predefined pools. No LLM needed."""
        pool = NICHE_HASHTAGS.get(niche, NICHE_HASHTAGS["lifestyle"])

        high = pool["high_volume"]
        medium = pool["medium_volume"]
        niche_tags = pool["niche"]

        # Target mix: 5 high-volume + 10 medium + 10 niche
        selected = []
        selected.extend(random.sample(high, min(5, len(high))))
        selected.extend(random.sample(medium, min(10, len(medium))))
        selected.extend(random.sample(niche_tags, min(10, len(niche_tags))))

        # Add theme as hashtag if provided
        if theme:
            theme_tag = theme.lower().replace(" ", "").replace("-", "")
            if theme_tag not in selected:
                selected.insert(0, theme_tag)

        return selected[:count]

    def format_hashtags(self, tags: list[str]) -> str:
        """Format tags as space-separated hashtag string."""
        return " ".join(f"#{tag.lstrip('#')}" for tag in tags)
