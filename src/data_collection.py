"""Data collection module for digital media analytics.

Provides YouTubeDataCollector (YouTube Data API v3) and SyntheticDataCollector
(fallback when no API key is available). Both produce standardised DataFrames
for channels, videos, comments, and comment replies.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inline configuration (mirrors key values from __init__.py)
# ---------------------------------------------------------------------------

CONFIG = {
    "CHANNELS": [
    {"name": "Aaj TV (Aaj News)", "channel_id": "UCgBAPAcLsh_MAPvJprIz89w"},
    {"name": "Hum TV",             "channel_id": "UCEeEQxm6qc_qaTE7qTV5aLQ"},
    {"name": "Raftar",             "channel_id": "UC6zIImBjDqtEsVZfQLPoQSw"},
    {"name": "Har Pal Geo",        "channel_id": "UC62pPLZqx8JIrtW1kT5NPWA"},  # ← add this
],
    "YOUTUBE_API_KEY": "YOUR_YOUTUBE_API_KEY_HERE",
    "RAW_DATA_DIR": "./data/raw",
    "PROCESSED_DATA_DIR": "./data/processed",
    "SYNTHETIC_DATA_DIR": "./data/synthetic",
    "MAX_VIDEOS_PER_CHANNEL": 500,
    "MAX_COMMENTS_PER_VIDEO": 200,
    "RANDOM_STATE": 42,
}

# Ensure data directories exist
for _key in ["RAW_DATA_DIR", "PROCESSED_DATA_DIR", "SYNTHETIC_DATA_DIR"]:
    os.makedirs(CONFIG[_key], exist_ok=True)

# ---------------------------------------------------------------------------
# Realistic YouTube comment templates (100+)
# ---------------------------------------------------------------------------

COMMENT_TEMPLATES: list[str] = [
    # News / current-affairs — positive
    "Great reporting as always. Keep it up!",
    "Finally someone covering this issue properly. Thank you Aaj TV.",
    "Very informative segment. Learned a lot from this.",
    "Excellent journalism. This is what real news looks like.",
    "The anchor asked all the right questions. Well done.",
    "This analysis is spot on. Subscribed!",
    "Best news channel in Pakistan hands down.",
    "Quality journalism still exists, and this channel proves it.",
    "Thank you for bringing this to light. Keep up the good work.",
    "Balanced reporting. Refreshing to see in today's media landscape.",
    "The investigative work here is commendable.",
    "This is the kind of content we need more of.",
    "Watching from London. Best source for Pakistani news.",
    "Finally, unbiased coverage of this issue.",
    "Well-researched and professionally presented.",
    # News / current-affairs — negative
    "Biased reporting as usual. Disappointed.",
    "Why don't you cover the real issues affecting common people?",
    "Clickbait title. Content doesn't match at all.",
    "Same old stories recycled every week.",
    "The anchor keeps interrupting the guest. Let them speak!",
    "This channel has become too sensationalist lately.",
    "Fake news spreaders. Do some actual journalism.",
    "Stop giving airtime to these corrupt politicians.",
    "Where is the follow-up on the previous investigation?",
    "Too much drama, not enough facts.",
    "Poor audio quality ruins an otherwise good segment.",
    "The ticker is distracting. Please remove it.",
    "Why was my previous comment deleted? Censorship much?",
    "You guys have lost credibility over the years.",
    "Report on education and healthcare for once, not just politics.",
    # News / current-affairs — neutral
    "What time does the bulletin air in UAE?",
    "Can someone summarise the main points?",
    "Interesting discussion. Would love to see the full debate.",
    "Is there a follow-up segment planned for this?",
    "The graphics in this segment are well done.",
    "Could you provide a source for that statistic?",
    "First time watching. What's the channel schedule?",
    "Good discussion but I wish it was longer.",
    "Neutral take on a complex issue. Appreciated.",
    "Anyone else watching in 2024?",
    "The background music is a bit loud.",
    "What is the name of the guest speaker?",
    "Looking forward to the next episode.",
    "Decent coverage. Room for improvement though.",
    "Aaj TV or Geo — which one do you all prefer?",
    # Entertainment / Hum TV — positive
    "This drama is so addictive! Best serial of the year.",
    "The chemistry between the leads is phenomenal.",
    "Mahira Khan is absolutely stunning in this role.",
    "OST of this drama is on repeat. Beautiful composition.",
    "The writing is brilliant. Every episode leaves you wanting more.",
    "Pakistani dramas are way ahead of Indian serials in quality.",
    "This scene gave me goosebumps. Outstanding acting.",
    "The director deserves all the awards for this masterpiece.",
    "Can't wait for the next episode. The suspense is killing me!",
    "Watching from Canada. These dramas remind me of home.",
    "Fawad Khan's performance is world-class.",
    "The cinematography in this episode was film-level.",
    "Hum TV never disappoints. Quality content always.",
    "This storyline is so relatable. Happens in every family.",
    "Cried my eyes out watching this episode. So emotional.",
    # Entertainment / Hum TV — negative
    "The story is dragging too much now. End it already.",
    "Same plot in every drama — toxic in-laws and crying bahu.",
    "Why does every drama have to be 40+ episodes? Keep it short.",
    "The FL's character is so weak. Terrible writing.",
    "Overacting at its peak in this scene.",
    "Predictable storyline. Called the ending in episode 5.",
    "Stop glorifying toxic relationships in the name of love.",
    "The makeup is too heavy. Doesn't look natural at all.",
    "Why kill off the only good character? Makes no sense.",
    "Soundtrack is good but the drama is a letdown.",
    "Product placement in every other scene ruins the immersion.",
    "The pacing in recent episodes has been terrible.",
    "Wasted potential. The premise was so good.",
    "No character development whatsoever after 20 episodes.",
    "Hum TV's quality has declined compared to the golden era.",
    # Entertainment / Hum TV — neutral
    "Which drama is better, this one or Mere Humsafar?",
    "Does anyone know the name of the actress playing the sister?",
    "What day does this drama air? New to the channel.",
    "The OST name please?",
    "Is this available with English subtitles anywhere?",
    "How many total episodes are planned for this serial?",
    "The wardrobe styling in this drama is on point.",
    "Decent drama. Watchable with family.",
    "Not the best but not the worst either.",
    "Pacing is slow but the story is engaging.",
    "Is this based on a novel? Feels like it.",
    "Watching all episodes on YouTube. Binge mode activated.",
    "The child actor in this drama is adorable.",
    "Can someone tell me where this was filmed? Beautiful location.",
    "Old Hum TV dramas were something else. Nostalgia.",
    # Music / Raftar — positive
    "Raftar never misses. Another banger!",
    "This beat is fire! Production quality through the roof.",
    "Lyrics hit different at 2am. Pure art.",
    "Desi hip-hop is evolving and Raftar is leading the charge.",
    "The flow switch in the second verse was insane.",
    "How does this not have millions of views yet? Underrated gem.",
    "Raftar x Kr$na collab when? The world needs it.",
    "This song has been on repeat for a week straight.",
    "The music video is a visual treat. Cinematic excellence.",
    "Finally, someone representing the streets authentically.",
    "Raftar's pen game is unmatched in the scene.",
    "The hook is so catchy. Can't get it out of my head.",
    "Goosebumps when the beat drops. Every. Single. Time.",
    "Real hip-hop finally getting recognition in South Asia.",
    "This deserves a billion views. Sharing everywhere.",
    # Music / Raftar — negative
    "Same flow in every song. Getting repetitive.",
    "Old Raftar was better. Commercial success ruined the art.",
    "Too much auto-tune. Can barely hear the actual vocals.",
    "The lyrics have gotten weaker compared to earlier work.",
    "Disappointing. Expected more from someone of his caliber.",
    "Why is every song now about money and cars? No substance.",
    "The mix is muddy. Can't hear the lyrics over the bass.",
    "Overhyped. There are better underground artists out there.",
    "Music video budget was clearly low. Looks amateur.",
    "Features the same 3 artists in every track. Branch out.",
    "This is just noise. Bring back meaningful rap.",
    "Sellout vibes from this one. Sold his art for views.",
    "The diss culture in DHH is cringe. Make music, not drama.",
    "Four minutes of the same bar repeated. Creativity where?",
    "Unpopular opinion: Raftar is overrated.",
    # Music / Raftar — neutral
    "Not his best work but not bad either.",
    "Who produced this beat? It's really good.",
    "Anyone have the lyrics written out?",
    "Raftar or Divine — who's the real G.O.A.T?",
    "When is the album dropping? Been waiting forever.",
    "The visuals are cool but the song is mid.",
    "Is this the official audio or a leak?",
    "What's the sample used in this track?",
    "Listening from Mumbai. Representing DHH worldwide.",
    "Good gym track. Added to my workout playlist.",
    "The feature artist outshined Raftar on this one, ngl.",
    "Decent track. Hope the album is better.",
    "This is growing on me after a few listens.",
    "Can someone recommend similar artists?",
    "The genre fusion in this one is interesting.",
    # Urdu / Roman Urdu mixed comments
    "Kamaal kar diya. Zabardast reporting.",
    "Bohot khoob. Aise hi mehnat karte raho.",
    "Yeh drama dekh kar rona aa gaya. Bohat emotional tha.",
    "Maza nahi aaya. Pehle behtar tha yeh channel.",
    "Kya bakwas hai yeh. Time waste.",
    "Allah khair kare. Kya ho raha hai mulk mein.",
    "Shabash beta. Bohat achi coverage thi.",
    "Yaar yeh song sun kar mood fresh ho gaya.",
    "Bhai kya gaana hai. Repeat pe sun raha hoon.",
    "Ajj kal ke dramon mein woh baat nahi rahi.",
    "Kya baat hai janab. Dil khush kar diya.",
    "Umeed hai agla episode bhi aisa hi hoga.",
    "Fazool reporting. Kuch naya batao.",
    "Wah wah. Kya scene hai. Maza aa gaya.",
    "Bilkul sahi baat ki hai aapne.",
    "Inshallah yeh drama hit hoga.",
    "Mashallah bohat acha kaam kar rahe ho.",
    "Aray wah! Zabardast performance.",
    "Bhai nay to kamaal kar diya.",
    "Kuch samajh nahi aaya lekin sun kar acha laga.",
    "Yeh dekh kar dil dukha. Afsos hua.",
    "Bohat khoobsurat awaaz hai.",
    "Mera pasandeeda drama hai yeh.",
    "Kahan ho bhai log? Like karo is comment ko!",
    "Acha laga. Keep it up!",
]


def _seed_random() -> None:
    random.seed(CONFIG["RANDOM_STATE"])
    np.random.seed(CONFIG["RANDOM_STATE"])


# ===================================================================
# YouTubeDataCollector
# ===================================================================


class YouTubeDataCollector:
    """Collects data from YouTube Data API v3.

    Parameters
    ----------
    api_key : str
        YouTube Data API v3 key.
    channels : list[dict]
        List of dicts with ``name`` and ``channel_id`` keys.
    """

    def __init__(self, api_key: str, channels: list[dict[str, str]]) -> None:
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        self.api_key = api_key
        self.channels = channels
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self._request_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _parse_duration(iso_duration: str) -> int | None:
        """Convert ISO 8601 duration to total seconds (or None)."""
        import re

        if not iso_duration:
            return None
        pattern = re.compile(
            r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?"
            r"T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        )
        m = pattern.match(iso_duration)
        if not m:
            return None
        parts = m.groups()
        multipliers = [31536000, 2592000, 604800, 86400, 3600, 60, 1]
        total = 0
        for val, mult in zip(parts, multipliers):
            if val:
                total += int(val) * mult
        return total

    def _backoff(self, attempt: int) -> None:
        delay = min(2 ** attempt + random.uniform(0, 1), 60)
        time.sleep(delay)

    # ------------------------------------------------------------------
    # Channel metadata
    # ------------------------------------------------------------------

    def collect_channel_metadata(self, channel_id: str) -> dict[str, Any]:
        """Return dict of channel statistics."""
        for attempt in range(5):
            try:
                resp = (
                    self.youtube.channels()
                    .list(part="snippet,statistics", id=channel_id)
                    .execute()
                )
                self._request_count += 1
                items = resp.get("items", [])
                if not items:
                    return {}
                snippet = items[0].get("snippet", {})
                stats = items[0].get("statistics", {})
                return {
                    "channel_id": channel_id,
                    "channel_name": snippet.get("title"),
                    "subscriber_count": self._safe_int(stats.get("subscriberCount")),
                    "view_count": self._safe_int(stats.get("viewCount")),
                    "video_count": self._safe_int(stats.get("videoCount")),
                    "country": snippet.get("country"),
                }
            except Exception as exc:
                err_str = str(exc).lower()
                if "quota" in err_str or "403" in err_str:
                    raise
                if attempt < 4:
                    self._backoff(attempt)
                else:
                    raise
        return {}

    # ------------------------------------------------------------------
    # Video list
    # ------------------------------------------------------------------

    def collect_video_list(
        self, channel_id: str, max_results: int = 200
    ) -> list[dict[str, Any]]:
        """Page through search + videos list and return video metadata."""
        channel_name = ""
        for ch in self.channels:
            if ch["channel_id"] == channel_id:
                channel_name = ch["name"]
                break

        video_ids: list[str] = []

        # --- search list to get video IDs ---
        page_token: str | None = ""
        while page_token is not None and len(video_ids) < max_results:
            for attempt in range(5):
                try:
                    kwargs: dict[str, Any] = {
                        "part": "id",
                        "channelId": channel_id,
                        "maxResults": min(50, max_results - len(video_ids)),
                        "type": "video",
                        "order": "date",
                    }
                    if page_token:
                        kwargs["pageToken"] = page_token
                    search_resp = self.youtube.search().list(**kwargs).execute()
                    self._request_count += 1
                    page_token = search_resp.get("nextPageToken")
                    for item in search_resp.get("items", []):
                        vid = item.get("id", {}).get("videoId")
                        if vid:
                            video_ids.append(vid)
                    break
                except Exception:
                    if attempt < 4:
                        self._backoff(attempt)
                    else:
                        page_token = None
                        break

        if not video_ids:
            return []

        # --- videos list in batches to get full metadata ---
        records: list[dict[str, Any]] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            for attempt in range(5):
                try:
                    v_resp = (
                        self.youtube.videos()
                        .list(
                            part="snippet,statistics,contentDetails,topicDetails",
                            id=",".join(batch),
                        )
                        .execute()
                    )
                    self._request_count += 1
                    for item in v_resp.get("items", []):
                        s = item.get("snippet", {})
                        stats = item.get("statistics", {})
                        cd = item.get("contentDetails", {})
                        td = item.get("topicDetails", {})
                        thumb = (
                            s.get("thumbnails", {})
                            .get("high", {})
                            .get("url")
                            or s.get("thumbnails", {})
                            .get("medium", {})
                            .get("url")
                            or s.get("thumbnails", {})
                            .get("default", {})
                            .get("url")
                        )
                        tags = s.get("tags", [])
                        records.append(
                            {
                                "video_id": item.get("id"),
                                "channel_id": s.get("channelId", channel_id),
                                "channel_name": s.get("channelTitle", channel_name),
                                "title": s.get("title"),
                                "description": s.get("description"),
                                "published_at": s.get("publishedAt"),
                                "view_count": self._safe_int(stats.get("viewCount")),
                                "like_count": self._safe_int(stats.get("likeCount")),
                                "comment_count": self._safe_int(
                                    stats.get("commentCount")
                                ),
                                "duration_seconds": self._parse_duration(
                                    cd.get("duration")
                                ),
                                "tags": ",".join(tags) if tags else None,
                                "category_id": self._safe_int(s.get("categoryId")),
                                "topic_categories": (
                                    ",".join(td.get("topicCategories", []))
                                    if td.get("topicCategories")
                                    else None
                                ),
                                "thumbnail_url": thumb,
                            }
                        )
                    break
                except Exception:
                    if attempt < 4:
                        self._backoff(attempt)
                    else:
                        break
        return records

    # ------------------------------------------------------------------
    # Comment threads
    # ------------------------------------------------------------------

    def collect_comment_threads(
        self, video_id: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch top-level comment threads for a video."""
        results: list[dict[str, Any]] = []
        page_token: str | None = ""
        while page_token is not None and len(results) < max_results:
            for attempt in range(5):
                try:
                    kwargs: dict[str, Any] = {
                        "part": "snippet,replies",
                        "videoId": video_id,
                        "maxResults": min(100, max_results - len(results)),
                        "textFormat": "plainText",
                    }
                    if page_token:
                        kwargs["pageToken"] = page_token
                    resp = self.youtube.commentThreads().list(**kwargs).execute()
                    self._request_count += 1
                    page_token = resp.get("nextPageToken")
                    for item in resp.get("items", []):
                        top = item.get("snippet", {}).get("topLevelComment", {})
                        s = top.get("snippet", {})
                        reply_count = item.get("snippet", {}).get(
                            "totalReplyCount", 0
                        )
                        results.append(
                            {
                                "comment_id": top.get("id"),
                                "video_id": s.get("videoId", video_id),
                                "parent_comment_id": None,
                                "author_channel_id": s.get("authorChannelId", {}).get(
                                    "value"
                                ),
                                "author_name": s.get("authorDisplayName"),
                                "comment_text": s.get("textDisplay"),
                                "like_count": self._safe_int(s.get("likeCount")),
                                "published_at": s.get("publishedAt"),
                                "updated_at": s.get("updatedAt"),
                                "reply_count": reply_count,
                                "is_reply": False,
                            }
                        )
                    break
                except Exception:
                    if attempt < 4:
                        self._backoff(attempt)
                    else:
                        page_token = None
                        break
        return results

    # ------------------------------------------------------------------
    # Comment replies
    # ------------------------------------------------------------------

    def collect_comment_replies(
        self, parent_comment_id: str
    ) -> list[dict[str, Any]]:
        """Fetch replies to a parent comment."""
        results: list[dict[str, Any]] = []
        page_token: str | None = ""
        while page_token is not None:
            for attempt in range(5):
                try:
                    kwargs: dict[str, Any] = {
                        "part": "snippet",
                        "parentId": parent_comment_id,
                        "maxResults": 100,
                        "textFormat": "plainText",
                    }
                    if page_token:
                        kwargs["pageToken"] = page_token
                    resp = self.youtube.comments().list(**kwargs).execute()
                    self._request_count += 1
                    page_token = resp.get("nextPageToken")
                    for item in resp.get("items", []):
                        s = item.get("snippet", {})
                        results.append(
                            {
                                "comment_id": item.get("id"),
                                "video_id": s.get("videoId"),
                                "parent_comment_id": parent_comment_id,
                                "author_channel_id": s.get(
                                    "authorChannelId", {}
                                ).get("value"),
                                "author_name": s.get("authorDisplayName"),
                                "comment_text": s.get("textDisplay"),
                                "like_count": self._safe_int(s.get("likeCount")),
                                "published_at": s.get("publishedAt"),
                                "updated_at": s.get("updatedAt"),
                                "reply_count": 0,
                                "is_reply": True,
                            }
                        )
                    break
                except Exception:
                    if attempt < 4:
                        self._backoff(attempt)
                    else:
                        page_token = None
                        break
        return results

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def collect_all(self) -> tuple[pd.DataFrame, ...]:
        """Run full collection and return (channels_df, videos_df, comments_df, replies_df)."""
        channels_data: list[dict] = []
        videos_data: list[dict] = []
        comments_data: list[dict] = []
        replies_data: list[dict] = []

        for ch in self.channels:
            cid = ch["channel_id"]
            print(f"[API] Collecting channel metadata: {ch['name']}  ({cid})")
            meta = self.collect_channel_metadata(cid)
            if meta:
                channels_data.append(meta)

            print(
                f"[API] Collecting videos for: {ch['name']} "
                f"(max {CONFIG['MAX_VIDEOS_PER_CHANNEL']})"
            )
            videos = self.collect_video_list(
                cid, max_results=CONFIG["MAX_VIDEOS_PER_CHANNEL"]
            )
            videos_data.extend(videos)

            for v in videos:
                vid = v["video_id"]
                print(
                    f"  [API] Comments for video: {vid[:20]}... "
                    f"(max {CONFIG['MAX_COMMENTS_PER_VIDEO']})"
                )
                threads = self.collect_comment_threads(
                    vid, max_results=CONFIG["MAX_COMMENTS_PER_VIDEO"]
                )
                comments_data.extend(threads)

                for t in threads:
                    if t["reply_count"] and t["reply_count"] > 0:
                        replies = self.collect_comment_replies(t["comment_id"])
                        replies_data.extend(replies)

        return (
            pd.DataFrame(channels_data),
            pd.DataFrame(videos_data),
            pd.DataFrame(comments_data),
            pd.DataFrame(replies_data),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_raw_json(self, data: Any, filename: str) -> None:
        path = os.path.join(CONFIG["RAW_DATA_DIR"], filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"[API] Saved raw JSON  ->  {path}")

    @staticmethod
    def save_parquet(df: pd.DataFrame, filename: str) -> None:
        path = os.path.join(CONFIG["PROCESSED_DATA_DIR"], filename)
        df.to_parquet(path, index=False)
        print(f"[API] Saved parquet  ->  {path}")


# ===================================================================
# SyntheticDataCollector
# ===================================================================


class SyntheticDataCollector:
    """Generates realistic synthetic YouTube data for development & testing."""

    def __init__(self) -> None:
        _seed_random()
        self.channels = CONFIG["CHANNELS"]
        self.max_videos = CONFIG["MAX_VIDEOS_PER_CHANNEL"]
        self.max_comments = CONFIG["MAX_COMMENTS_PER_VIDEO"]
        self.rng = np.random.default_rng(CONFIG["RANDOM_STATE"])

    # ------------------------------------------------------------------
    # Channel metadata
    # ------------------------------------------------------------------

    def _generate_channels(self) -> pd.DataFrame:
        base_subs = [3_500_000, 5_200_000, 1_800_000]
        base_views = [850_000_000, 2_100_000_000, 420_000_000]
        rows = []
        for ch, subs, views in zip(self.channels, base_subs, base_views):
            rows.append(
                {
                    "channel_id": ch["channel_id"],
                    "channel_name": ch["name"],
                    "subscriber_count": int(subs * (1 + self.rng.normal(0, 0.03))),
                    "view_count": int(views * (1 + self.rng.normal(0, 0.05))),
                    "video_count": int(
                        1500 + self.rng.integers(-200, 200)
                    ),
                    "country": "PK",
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Video metadata
    # ------------------------------------------------------------------

    _video_titles: dict[str, list[str]] = {
        "Aaj TV (Aaj News)": [
            "Breaking News: Major Political Development in Islamabad",
            "Budget 2024: Key Highlights and Analysis",
            "Exclusive Interview with Prime Minister on Economic Reforms",
            "Monsoon Rains Wreak Havoc Across Sindh",
            "Supreme Court Verdict on Reserved Seats Case",
            "Education Crisis: Out-of-School Children in Pakistan",
            "Pakistan Cricket Team's World Cup Journey",
            "Climate Change Impact on Agriculture in Punjab",
            "Election 2024: Voter Turnout and Results Analysis",
            "Terrorism Resurgence in KP: A Ground Report",
            "Digital Pakistan: IT Exports Hit New Record",
            "Energy Crisis: Load-shedding Returns to Urban Centers",
            "Women's Rights Bill Passed in National Assembly",
            "Inflation Hits Double Digits: How Are People Coping?",
            "CPEC Phase 2: New Projects Announced",
            "Pakistan-Afghanistan Border Tensions Explained",
            "Healthcare Crisis: Public Hospitals in Distress",
            "Gwadar Port Development: Progress Report",
            "Pakistani Students Win International Robotics Competition",
            "Cultural Heritage: Preserving Mohenjo-Daro",
        ],
        "Hum TV": [
            "Mere Humsafar Episode 25 Full Drama",
            "Tere Bin Episode 42 Last Episode Promo",
            "Ishq Murshid Episode 13 Full HD",
            "Suno Chanda Season 2 Episode 1",
            "Zindagi Gulzar Hai Episode 15 Classic Scene",
            "Ehd-e-Wafa Episode 10 Best Moments",
            "Mere Paas Tum Ho Episode 20 Emotional Climax",
            "Parizaad Episode 28: The Transformation",
            "Yakeen Ka Safar Episode 8 Heart-Touching Dialogues",
            "Khaani Episode 18: Mir Hadi's Confrontation",
            "Raqs-e-Bismil Episode 22: The Dance Sequence",
            "Alif Episode 17: Momin's Spiritual Journey",
            "Diyar-e-Dil Episode 12: Family Reunion",
            "Udaari Episode 15: Breaking the Silence",
            "Dunk Episode 8: Courtroom Drama",
            "Ranjha Ranjha Kardi Episode 30 Final Episode",
            "Sinf-e-Aahan Episode 6: Training Montage",
            "Hum Kahan Ke Sachay Thay Episode 14",
            "Chupke Chupke Episode 1: Ramazan Special",
            "Aangan Episode 5: Partition Era Storytelling",
        ],
        "Raftar": [
            "Raftar - GOAT Flow (Official Music Video)",
            "Raftar x Kr$na - Saza-E-Maut | Full Audio",
            "Raftar - Black Sheep | Official Music Video",
            "Raftar - PRAA | Reaction & Breakdown",
            "Raftar - Ghana Kasoota | Dance Video",
            "Raftar - Microphone Check | Old School Vibe",
            "Raftar - Aage Chal | Motivational Rap",
            "Raftar ft. Badshah - Bandana Gang | Lyrical Video",
            "Raftar - Toh Kya | Kalamkaar Presents",
            "Raftar - Jhakkas | Party Anthem 2024",
            "Raftar - Trap Praa | Hard Hitting Bars",
            "Raftar - Damn Son | Behind The Scenes",
            "Raftar - Mantoiyat | Poetic Rap",
            "Raftar - F16 | High Energy Performance",
            "Raftar - No China | Social Commentary",
            "Raftar - Saath Ya Khilaaf | Collab Track",
            "Raftar - Beshaq | Emotional Rap",
            "Raftar - Kaam 25 | Street Anthem",
            "Raftar - Ice | Cool Vibe Rap",
            "Raftar - Legacy | Career Retrospective",
        ],
    }

    def _generate_videos(self) -> pd.DataFrame:
        end_date = datetime(2026, 5, 1)
        start_date = end_date - timedelta(days=730)
        records = []
        for ch in self.channels:
            name = ch["name"]
            titles = self._video_titles.get(name, [f"Video {i}" for i in range(20)])
            for i in range(self.max_videos):
                days_offset = self.rng.integers(0, 730)
                pub = start_date + timedelta(days=int(days_offset))
                pub += timedelta(
                    hours=int(self.rng.integers(0, 23)),
                    minutes=int(self.rng.integers(0, 59)),
                    seconds=int(self.rng.integers(0, 59)),
                )
                base_views = max(
                    100, int(np.random.lognormal(mean=10.5, sigma=1.8))
                )
                engagement_rate = max(0.001, self.rng.normal(0.04, 0.015))
                likes = int(base_views * engagement_rate)
                comments = max(0, int(base_views * engagement_rate * 0.15))
                duration_pool = [180, 300, 420, 600, 900, 1200, 1800, 2400, 3600]
                duration = int(self.rng.choice(duration_pool))

                records.append(
                    {
                        "video_id": f"synth_{ch['channel_id'][:8]}_{i:04d}",
                        "channel_id": ch["channel_id"],
                        "channel_name": name,
                        "title": self.rng.choice(titles)
                        + (f" (Part {i % 5 + 1})" if i % 3 == 0 else ""),
                        "description": (
                            f"Watch this video from {name}. "
                            f"Subscribe for more content. "
                            f"#YouTube #Pakistan #Content"
                        ),
                        "published_at": pub.isoformat(),
                        "view_count": int(base_views),
                        "like_count": int(likes),
                        "comment_count": int(comments),
                        "duration_seconds": duration,
                        "tags": "pakistan,entertainment,trending,viral",
                        "category_id": self.rng.choice([24, 25, 10, 22, 23]),
                        "topic_categories": (
                            "https://en.wikipedia.org/wiki/Entertainment,"
                            "https://en.wikipedia.org/wiki/Music"
                        ),
                        "thumbnail_url": (
                            f"https://picsum.photos/seed/{ch['channel_id'][:4]}"
                            f"{i}/480/360"
                        ),
                    }
                )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Comments & replies
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_comment_text(channel_name: str) -> str:
        """Pick a template biased toward the channel's genre."""
        is_news = "News" in channel_name or "Aaj" in channel_name
        is_music = "Raftar" in channel_name or "Raftar" in channel_name.lower()
        if is_news:
            pool = COMMENT_TEMPLATES[:90]  # news + urdu
        elif is_music:
            pool = COMMENT_TEMPLATES[60:]
        else:
            pool = COMMENT_TEMPLATES  # Hum TV gets all
        return random.choice(pool).strip()

    @staticmethod
    def _generate_author(channel_name: str, idx: int) -> str:
        prefixes = [
            "Ahmed", "Fatima", "Hassan", "Ayesha", "Zainab",
            "Bilal", "Sana", "Usman", "Nadia", "Farhan",
            "Mehwish", "Ali", "Hira", "Saad", "Rabia",
            "Tariq", "Saima", "Kamran", "Amna", "Omar",
            "Zara", "Imran", "Bushra", "Danish", "Sadia",
            "Waqar", "Noor", "Asad", "Sara", "Hamza",
        ]
        prefix = prefixes[idx % len(prefixes)]
        return f"{prefix}_{channel_name.replace(' ', '')[:4]}{idx:04d}"

    def _generate_comments(
        self, videos_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        comment_records: list[dict] = []
        reply_records: list[dict] = []
        comment_counter = 0

        for _, video in videos_df.iterrows():
            vid = video["video_id"]
            ch_name = video["channel_name"]
            num_comments = min(
                self.max_comments,
                max(10, self.rng.poisson(lam=35)),
            )
            for ci in range(num_comments):
                cid = f"c_{vid}_{ci:03d}"
                pub_date = datetime.fromisoformat(video["published_at"]) + timedelta(
                    days=int(self.rng.integers(0, 60)),
                    hours=int(self.rng.integers(0, 23)),
                )
                comment_records.append(
                    {
                        "comment_id": cid,
                        "video_id": vid,
                        "parent_comment_id": None,
                        "author_channel_id": f"UC_synth_author_{comment_counter:05d}",
                        "author_name": self._generate_author(ch_name, comment_counter),
                        "comment_text": self._pick_comment_text(ch_name),
                        "like_count": max(0, int(self.rng.exponential(scale=3))),
                        "published_at": pub_date.isoformat(),
                        "updated_at": pub_date.isoformat(),
                        "reply_count": 0,
                        "is_reply": False,
                    }
                )
                comment_counter += 1
                # Generate 0-5 replies per comment
                num_replies = self.rng.poisson(lam=1.5)
                if num_replies > 0:
                    comment_records[-1]["reply_count"] = int(num_replies)
                for ri in range(int(num_replies)):
                    rid = f"r_{cid}_{ri:02d}"
                    r_pub = pub_date + timedelta(
                        hours=int(self.rng.integers(1, 48))
                    )
                    reply_records.append(
                        {
                            "comment_id": rid,
                            "video_id": vid,
                            "parent_comment_id": cid,
                            "author_channel_id": (
                                f"UC_synth_author_{comment_counter:05d}"
                            ),
                            "author_name": self._generate_author(
                                ch_name, comment_counter
                            ),
                            "comment_text": self._pick_comment_text(ch_name),
                            "like_count": max(
                                0, int(self.rng.exponential(scale=2))
                            ),
                            "published_at": r_pub.isoformat(),
                            "updated_at": r_pub.isoformat(),
                            "reply_count": 0,
                            "is_reply": True,
                        }
                    )
                    comment_counter += 1

        comments_df = pd.DataFrame(comment_records)
        replies_df = pd.DataFrame(reply_records)
        return comments_df, replies_df

    # ------------------------------------------------------------------
    # Time-series snapshots
    # ------------------------------------------------------------------

    def _generate_timeseries(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """Generate multiple observations per video across time."""
        snapshots: list[dict] = []
        observation_dates = pd.date_range(
            start="2024-05-01", end="2026-05-01", freq="7D"
        )
        for _, video in videos_df.iterrows():
            pub_date = datetime.fromisoformat(video["published_at"])
            base_views = video["view_count"]
            base_likes = video["like_count"] or 0
            base_comments = video["comment_count"] or 0
            for obs in observation_dates:
                if obs.to_pydatetime() < pub_date:
                    continue
                days_since = (obs.to_pydatetime() - pub_date).days
                # Logistic growth curve for cumulative metrics
                growth = 1 / (1 + np.exp(-0.01 * (days_since - 30))) + self.rng.normal(
                    0, 0.02
                )
                growth = max(0, min(1.2, growth))
                snapshots.append(
                    {
                        "video_id": video["video_id"],
                        "observation_date": obs.strftime("%Y-%m-%d"),
                        "cumulative_views": int(base_views * growth),
                        "cumulative_likes": int(base_likes * growth),
                        "cumulative_comments": int(base_comments * growth),
                        "days_since_published": days_since,
                    }
                )
        return pd.DataFrame(snapshots)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def collect_all(self) -> tuple[pd.DataFrame, ...]:
        """Generate synthetic datasets and return 5 DataFrames."""
        print("[Synthetic] Generating channel metadata ...")
        channels_df = self._generate_channels()
        print(f"[Synthetic]   -> {len(channels_df)} channels")

        print("[Synthetic] Generating video metadata ...")
        videos_df = self._generate_videos()
        print(f"[Synthetic]   -> {len(videos_df)} videos")

        print("[Synthetic] Generating comments & replies ...")
        comments_df, replies_df = self._generate_comments(videos_df)
        print(
            f"[Synthetic]   -> {len(comments_df)} comments, "
            f"{len(replies_df)} replies"
        )

        print("[Synthetic] Generating time-series snapshots ...")
        timeseries_df = self._generate_timeseries(videos_df)
        print(f"[Synthetic]   -> {len(timeseries_df)} snapshot rows")

        return channels_df, videos_df, comments_df, replies_df, timeseries_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_parquet(df: pd.DataFrame, filename: str) -> None:
        path = os.path.join(CONFIG["SYNTHETIC_DATA_DIR"], filename)
        df.to_parquet(path, index=False)
        print(f"[Synthetic] Saved parquet  ->  {path}")


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    api_key = CONFIG["YOUTUBE_API_KEY"]
    use_real_api = api_key and api_key not in ("YOUR_YOUTUBE_API_KEY_HERE", "", None)
    channels_df: pd.DataFrame
    videos_df: pd.DataFrame
    comments_df: pd.DataFrame
    replies_df: pd.DataFrame
    timeseries_df: pd.DataFrame

    if use_real_api:
        try:
            print("=" * 60)
            print(" Using YouTube Data API v3 collector")
            print("=" * 60)
            collector = YouTubeDataCollector(api_key, CONFIG["CHANNELS"])
            channels_df, videos_df, comments_df, replies_df = collector.collect_all()

            collector.save_raw_json(
                channels_df.to_dict(orient="records"), "channels.json"
            )
            collector.save_raw_json(
                videos_df.to_dict(orient="records"), "videos.json"
            )
            collector.save_raw_json(
                comments_df.to_dict(orient="records"), "comments.json"
            )
            collector.save_raw_json(
                replies_df.to_dict(orient="records"), "replies.json"
            )

            collector.save_parquet(channels_df, "channels.parquet")
            collector.save_parquet(videos_df, "videos.parquet")
            collector.save_parquet(comments_df, "comments.parquet")
            collector.save_parquet(replies_df, "comment_replies.parquet")

            # Also generate a timeseries from the collected data
            syn = SyntheticDataCollector()
            timeseries_df = syn._generate_timeseries(videos_df)
            syn.save_parquet(timeseries_df, "video_statistics_timeseries.parquet")

            print("\n" + "=" * 60)
            print(" API collection complete.")
            print(f"   Channels:   {len(channels_df)}")
            print(f"   Videos:     {len(videos_df)}")
            print(f"   Comments:   {len(comments_df)}")
            print(f"   Replies:    {len(replies_df)}")
            print(f"   Timeseries: {len(timeseries_df)}")
            print(f"   API calls:  {collector._request_count}")
            print("=" * 60)

        except Exception as exc:
            print(f"[ERROR] API collection failed: {exc}")
            print("[INFO]  Falling back to synthetic data generation ...")
            use_real_api = False

    if not use_real_api:
        print("=" * 60)
        print(" Using Synthetic data collector (fallback)")
        print("=" * 60)
        syn = SyntheticDataCollector()
        (
            channels_df,
            videos_df,
            comments_df,
            replies_df,
            timeseries_df,
        ) = syn.collect_all()

        syn.save_parquet(channels_df, "channels.parquet")
        syn.save_parquet(videos_df, "videos.parquet")
        syn.save_parquet(comments_df, "comments.parquet")
        syn.save_parquet(replies_df, "comment_replies.parquet")
        syn.save_parquet(timeseries_df, "video_statistics_timeseries.parquet")

        print("\n" + "=" * 60)
        print(" Synthetic data generation complete.")
        print(f"   Channels:   {len(channels_df)}")
        print(f"   Videos:     {len(videos_df)}")
        print(f"   Comments:   {len(comments_df)}")
        print(f"   Replies:    {len(replies_df)}")
        print(f"   Timeseries: {len(timeseries_df)}")
        print("=" * 60)
