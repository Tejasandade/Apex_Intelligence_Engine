from typing import List

class NewsFilter:
    """
    Information filtration pipeline to discard noise and isolate 
    high-impact news related to specific assets.
    """
    
    def __init__(self):
        # Common spam, promotional, or opinion keywords to filter out
        self.noise_keywords = [
            "giveaway", "airdrop", "sponsored", "opinion",
            "promoted", "presale", "bonus", "referral", "win", "free"
        ]
        
    def filter_by_asset(self, headlines: List[str], keywords: List[str]) -> List[str]:
        """
        Keeps only headlines that contain at least one of the asset keywords.
        """
        filtered = []
        for headline in headlines:
            headline_lower = headline.lower()
            if any(keyword.lower() in headline_lower for keyword in keywords):
                filtered.append(headline)
        return filtered

    def remove_noise(self, headlines: List[str]) -> List[str]:
        """
        Discards headlines containing spam, opinion pieces, or irrelevant promotional terms.
        """
        clean_headlines = []
        for headline in headlines:
            headline_lower = headline.lower()
            if not any(noise.lower() in headline_lower for noise in self.noise_keywords):
                clean_headlines.append(headline)
        return clean_headlines
