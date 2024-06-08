import datetime
import os
from typing import Any, NamedTuple

import isodate
import requests


class Video(NamedTuple):
    id: str
    channel_id: str
    title: str
    description: str
    duration: int
    published_at: str


class Channel(NamedTuple):
    channel_id: str
    channel_name: str
    updated_at: int


class TopLevelComment(NamedTuple):
    video_id: str
    comment_id: str
    text: str


class SearchedResultVideo(NamedTuple):
    id: str
    channel_id: str
    title: str
    description: str
    published_at: str


class SearchResult(NamedTuple):
    searched_at: str
    q: str
    result: list[SearchedResultVideo]


def fetch_videos(video_ids: list[str]) -> list[Video]:
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    param = {
        "key": os.environ["YOUTUBE_DATA_API_KEY"],
        "id": ",".join(video_ids),
        "part": "snippet, contentDetails",
    }

    req = requests.get(video_url, params=param)
    result = req.json()

    return [
        Video(
            id=item["id"],
            channel_id=item["snippet"]["channelId"],
            title=item["snippet"]["title"],
            description=item["snippet"]["description"],
            duration=isodate.parse_duration(
                item["contentDetails"]["duration"]
            ).total_seconds(),
            published_at=item["snippet"]["publishedAt"],
        )
        for item in result["items"]
    ]


def fetch_video_statistics(video_ids: list[str]):
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    param = {
        "key": os.environ["YOUTUBE_DATA_API_KEY"],
        "id": ",".join(video_ids),
        "part": "statistics",
    }

    req = requests.get(video_url, params=param)
    result = req.json()
    print(result)

    return [{"id": item["id"], **item["statistics"]} for item in result["items"]]


def _call_channels(channel_id: str) -> dict[str, Any]:
    channel_url = "https://www.googleapis.com/youtube/v3/channels"
    param = {
        "key": os.environ["YOUTUBE_DATA_API_KEY"],
        "id": channel_id,
        "part": "snippet, contentDetails",
        "maxResults": "50",
    }

    req = requests.get(channel_url, params=param)
    return req.json()


def fetch_channel(channel_id: str) -> Channel:
    result = _call_channels(channel_id)
    item = result["items"][0]
    return Channel(
        channel_id=item["id"],
        channel_name=item["snippet"]["title"],
        updated_at=0,
    )


def fetch_uploads_playlist_id(channel_id: str) -> str:
    result = _call_channels(channel_id)
    item = result["items"][0]
    playlist_id = item["contentDetails"]["relatedPlaylists"]["uploads"]
    return playlist_id


def fetch_video_ids_from_playlist(playlist_id: str, pageToken: str):
    playlist_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    param = {
        "key": os.environ["YOUTUBE_DATA_API_KEY"],
        "playlistId": playlist_id,
        "part": "contentDetails",
        "maxResults": "50",
        "pageToken": pageToken,
    }

    req = requests.get(playlist_url, params=param)
    return req.json()


def fetch_video_ids_from_channel(channel_id: str) -> set[str]:
    # channel -> playlist
    playlist_id = fetch_uploads_playlist_id(channel_id)

    pageToken = ""
    video_ids: set[str] = set()
    while True:
        playlist_result = fetch_video_ids_from_playlist(playlist_id, pageToken)

        # 取得失敗してたら飛ばす
        if "items" not in playlist_result:
            break

        # 今までの集合と今回の集合をマージする
        for item in playlist_result["items"]:
            video_ids.add(item["contentDetails"]["videoId"])

        # 残りのアイテム数がmaxResultsを超えている場合はnextPageTokenが帰ってくる
        if "nextPageToken" in playlist_result:
            pageToken = playlist_result["nextPageToken"]
        else:
            break

    return video_ids


# ページング処理をして全て取得
def fetch_toplevel_comments(video_id, page_token=None) -> list[TopLevelComment]:
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": os.environ["YOUTUBE_DATA_API_KEY"],
        "maxResults": 100,
    }
    if page_token:
        params["pageToken"] = page_token

    response = requests.get(url, params=params)
    data = response.json()
    comments = [
        TopLevelComment(
            video_id,
            item["id"],
            item["snippet"]["topLevelComment"]["snippet"]["textOriginal"],
        )
        for item in data["items"]
    ]

    if "nextPageToken" in data:
        comments += fetch_toplevel_comments(video_id, data["nextPageToken"])

    return comments


def search_videos(query: str) -> SearchResult:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": os.environ["YOUTUBE_DATA_API_KEY"],
        "maxResults": 50,
    }

    response = requests.get(url, params=params)
    data = response.json()

    return SearchResult(
        searched_at=datetime.datetime.now().isoformat(),
        q=query,
        result=[
            SearchedResultVideo(
                id=item["id"]["videoId"],
                channel_id=item["snippet"]["channelId"],
                title=item["snippet"]["title"],
                description=item["snippet"]["description"],
                published_at=item["snippet"]["publishedAt"],
            )
            for item in data["items"]
        ],
    )
