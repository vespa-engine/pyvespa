# Video Search and Retrieval with Vespa and TwelveLabs[¶](#video-search-and-retrieval-with-vespa-and-twelvelabs)

In the following notebook, we will demonstrate how to leverage [TwelveLabs](https://www.twelvelabs.io/) `Marengo-retrieval-2.7` a SOTA multimodal embedding model to demonstrate a use case of video embeddings storage and semantic search retrieval using Vespa.ai.

The steps we will take in this notebook are:

1. Setup and configuration
1. Generate Attributes and Embeddings for 3 sample videos using the TwelveLabs python SDK.
1. Deploy the Vespa application to Vespa Cloud and Feed the Data
1. Perform a semantic search with hybrid multi-phase ranking on the videos
1. Review the results
1. Cleanup

All the steps that are needed to provision the Vespa application, including feeding the data, can be done by running this notebook. We have tried to make it easy for others to run this notebook, to create your own Video semantic search application using TwelveLabs models with Vespa.

## 1. Setup and Configuration[¶](#1-setup-and-configuration)

For reference, this is the Python version used for this notebook.

In \[1\]:

Copied!

```
!python --version
```

!python --version

```
Python 3.12.4
```

### 1.1 Install libraries[¶](#11-install-libraries)

Install the required Python dependencies from TwelveLabs python SDK and pyvespa python API.

In \[2\]:

Copied!

```
!pip3 install pyvespa vespacli twelvelabs pandas
```

!pip3 install pyvespa vespacli twelvelabs pandas

```
Requirement already satisfied: pyvespa in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (0.55.0)
Requirement already satisfied: vespacli in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (8.391.23)
Requirement already satisfied: twelvelabs in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (0.4.10)
Requirement already satisfied: pandas in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (2.2.2)
Requirement already satisfied: requests in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (2.32.3)
Requirement already satisfied: requests_toolbelt in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (1.0.0)
Requirement already satisfied: docker in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (7.1.0)
Requirement already satisfied: jinja2 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (3.1.4)
Requirement already satisfied: cryptography in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (43.0.3)
Requirement already satisfied: aiohttp in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (3.10.10)
Requirement already satisfied: httpx[http2] in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (0.28.1)
Requirement already satisfied: tenacity>=8.4.1 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (9.0.0)
Requirement already satisfied: typing_extensions in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (4.12.2)
Requirement already satisfied: python-dateutil in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (2.9.0.post0)
Requirement already satisfied: fastcore>=1.7.8 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (1.7.19)
Requirement already satisfied: lxml in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pyvespa) (5.3.0)
Requirement already satisfied: pydantic>=2.4.2 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from twelvelabs) (2.10.6)
Requirement already satisfied: numpy>=1.26.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pandas) (1.26.4)
Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pandas) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pandas) (2023.3)
Requirement already satisfied: packaging in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from fastcore>=1.7.8->pyvespa) (24.2)
Requirement already satisfied: anyio in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from httpx[http2]->pyvespa) (4.8.0)
Requirement already satisfied: certifi in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from httpx[http2]->pyvespa) (2025.1.31)
Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from httpx[http2]->pyvespa) (1.0.7)
Requirement already satisfied: idna in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from httpx[http2]->pyvespa) (3.10)
Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from httpcore==1.*->httpx[http2]->pyvespa) (0.14.0)
Requirement already satisfied: annotated-types>=0.6.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pydantic>=2.4.2->twelvelabs) (0.7.0)
Requirement already satisfied: pydantic-core==2.27.2 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from pydantic>=2.4.2->twelvelabs) (2.27.2)
Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from python-dateutil->pyvespa) (1.16.0)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from aiohttp->pyvespa) (2.4.0)
Requirement already satisfied: aiosignal>=1.1.2 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from aiohttp->pyvespa) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from aiohttp->pyvespa) (24.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from aiohttp->pyvespa) (1.4.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from aiohttp->pyvespa) (6.0.4)
Requirement already satisfied: yarl<2.0,>=1.12.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from aiohttp->pyvespa) (1.15.5)
Requirement already satisfied: cffi>=1.12 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from cryptography->pyvespa) (1.17.1)
Requirement already satisfied: urllib3>=1.26.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from docker->pyvespa) (2.3.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from requests->pyvespa) (3.4.1)
Requirement already satisfied: h2<5,>=3 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from httpx[http2]->pyvespa) (4.1.0)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from jinja2->pyvespa) (3.0.2)
Requirement already satisfied: pycparser in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from cffi>=1.12->cryptography->pyvespa) (2.22)
Requirement already satisfied: hyperframe<7,>=6.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from h2<5,>=3->httpx[http2]->pyvespa) (6.0.1)
Requirement already satisfied: hpack<5,>=4.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from h2<5,>=3->httpx[http2]->pyvespa) (4.0.0)
Requirement already satisfied: propcache>=0.2.0 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from yarl<2.0,>=1.12.0->aiohttp->pyvespa) (0.2.0)
Requirement already satisfied: sniffio>=1.1 in /opt/anaconda3/envs/vespa-env/lib/python3.12/site-packages (from anyio->httpx[http2]->pyvespa) (1.3.1)
```

Import all the required packages in this notebook.

In \[3\]:

Copied!

```
import os
import hashlib
import json

from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    FieldSet,
    SecondPhaseRanking,
    Function,
)

from vespa.deployment import VespaCloud
from vespa.io import VespaResponse, VespaQueryResponse

from twelvelabs import TwelveLabs
from twelvelabs.models.embed import EmbeddingsTask

import pandas as pd

from datetime import datetime
```

import os import hashlib import json from vespa.package import ( ApplicationPackage, Field, Schema, Document, HNSW, RankProfile, FieldSet, SecondPhaseRanking, Function, ) from vespa.deployment import VespaCloud from vespa.io import VespaResponse, VespaQueryResponse from twelvelabs import TwelveLabs from twelvelabs.models.embed import EmbeddingsTask import pandas as pd from datetime import datetime

### 1.2 Get a TwelveLabs API key[¶](#12-get-a-twelvelabs-api-key)

[Sign-up](https://auth.twelvelabs.io/u/signup) for TwelveLabs.

After logging in, navigate to your profile and get your [API key](https://playground.twelvelabs.io/dashboard/api-key). Copy it and paste it below.

The Free plan includes indexing of 600 mins of videos, which should be sufficient to explore the capabilities of the API.

In \[8\]:

Copied!

```
TL_API_KEY = os.getenv("TL_API_KEY") or input("Enter your TL_API key: ")
```

TL_API_KEY = os.getenv("TL_API_KEY") or input("Enter your TL_API key: ")

### 1.3 Sign-up for a Vespa Trial Account[¶](#13-sign-up-for-a-vespa-trial-account)

**Pre-requisite**:

- Spin-up a Vespa Cloud [Trial](https://vespa.ai/free-trial) account.
- Login to the account you just created and create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/).
- Save the tenant name.

### 1.4 Setup the tenant name and the application name[¶](#14-setup-the-tenant-name-and-the-application-name)

- Paste below the name of the tenant name.
- Give your application a name. Note that the name cannot have `-` or `_`.

In \[ \]:

Copied!

```
# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"
# Replace with your application name (does not need to exist yet)
application = "videosearch"
```

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Replace with your application name (does not need to exist yet)

application = "videosearch"

## 2. Generate Attributes and Embeddings for sample videos using TwelveLabs Embedding API[¶](#2-generate-attributes-and-embeddings-for-sample-videos-using-twelvelabs-embedding-api)

### 2.1 Generate attributes on the videos[¶](#21-generate-attributes-on-the-videos)

In this section, we will leverage the [Pegasus 1.2](https://docs.twelvelabs.io/v1.3/docs/concepts/models/pegasus) generative model to generate some attributes about our videos to store as part of the searchable information in Vespa. Attributes we want to store as part of the videos include:

- Keywords
- Summaries

For video samples, we are selecting the 3 videos in the array below from the [Internet Archive](https://archive.org/).

You can customize this code with the urls of your choice. Note that there are certain restrictions such as the resolution of the videos.

In \[10\]:

Copied!

```
VIDEO_URLs = [
    "https://archive.org/download/the-end-blue-sky-studios/The%20End%281080P_60FPS%29.ia.mp4",
    "https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4",
    "https://archive.org/download/The_Worm_in_the_Apple_Animation_Test/AnimationTest.mov",
]
```

VIDEO_URLs = [ "https://archive.org/download/the-end-blue-sky-studios/The%20End%281080P_60FPS%29.ia.mp4", "https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4", "https://archive.org/download/The_Worm_in_the_Apple_Animation_Test/AnimationTest.mov", ]

In order to generate text on the videos, the prerequisite is to upload the videos and index them. Let's first create an index below:

In \[11\]:

Copied!

```
# Spin-up session
client = TwelveLabs(api_key=TL_API_KEY)

# Generating Index Name
timestamp = int(datetime.now().timestamp())
index_name = "Vespa_" + str(timestamp)

# Create Index
print("Creating Index:" + index_name)
index = client.index.create(
    name=index_name,
    models=[
        {
            "name": "pegasus1.2",
            "options": ["visual", "audio"],
        }
    ],
    addons=["thumbnail"],  # Optional
)
print(f"Created index: id={index.id} name={index.name} models={index.models}")
```

# Spin-up session

client = TwelveLabs(api_key=TL_API_KEY)

# Generating Index Name

timestamp = int(datetime.now().timestamp()) index_name = "Vespa\_" + str(timestamp)

# Create Index

print("Creating Index:" + index_name) index = client.index.create( name=index_name, models=\[ { "name": "pegasus1.2", "options": ["visual", "audio"], } \], addons=["thumbnail"], # Optional ) print(f"Created index: id={index.id} name={index.name} models={index.models}")

```
Creating Index:Vespa_1752595622
Created index: id=68767ca6e01b53f51c3f2ac5 name=Vespa_1752595622 models=root=[Model(name='pegasus1.2', options=['visual', 'audio'], addons=None, finetuned=False)]
```

We can now upload the videos:

In \[12\]:

Copied!

```
# Capturing index id for upload
index_id = index.id

def on_task_update(task: EmbeddingsTask):
    print(f"  Status={task.status}")


for video_url in VIDEO_URLs:
    # Create a video indexing task
    task = client.task.create(index_id=index_id, url=video_url)
    print(f"Task created successfully! Task ID: {task.id}")
    status = task.wait_for_done(sleep_interval=10, callback=on_task_update)
    print(f"Indexing done: {status}")
    if task.status != "ready":
        raise RuntimeError(f"Indexing failed with status {task.status}")
    print(
        f"Uploaded {video_url}. The unique identifer of your video is {task.video_id}."
    )
```

# Capturing index id for upload

index_id = index.id def on_task_update(task: EmbeddingsTask): print(f" Status={task.status}") for video_url in VIDEO_URLs:

# Create a video indexing task

task = client.task.create(index_id=index_id, url=video_url) print(f"Task created successfully! Task ID: {task.id}") status = task.wait_for_done(sleep_interval=10, callback=on_task_update) print(f"Indexing done: {status}") if task.status != "ready": raise RuntimeError(f"Indexing failed with status {task.status}") print( f"Uploaded {video_url}. The unique identifer of your video is {task.video_id}." )

```
Task created successfully! Task ID: 68767caa47c93cd3ab1e4b05
  Status=pending
  Status=pending
  Status=pending
  Status=pending
  Status=ready
Indexing done: Task(id='68767caa47c93cd3ab1e4b05', created_at='2025-07-15T16:07:08.998Z', updated_at='2025-07-15T16:07:08.998Z', index_id='68767ca6e01b53f51c3f2ac5', video_id='68767caa47c93cd3ab1e4b05', status='ready', system_metadata={'filename': 'The End(1080P_60FPS).ia.mp4', 'duration': 34.667392, 'width': 1920, 'height': 1080}, hls=TaskHLS(video_url='', thumbnail_urls=[], status='PROCESSING', updated_at='2025-07-15T16:07:08.998Z'))
Uploaded https://archive.org/download/the-end-blue-sky-studios/The%20End%281080P_60FPS%29.ia.mp4. The unique identifer of your video is 68767caa47c93cd3ab1e4b05.
Task created successfully! Task ID: 68767ce06c4253f85f0820d0
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=indexing
  Status=ready
Indexing done: Task(id='68767ce06c4253f85f0820d0', created_at='2025-07-15T16:08:01.059Z', updated_at='2025-07-15T16:08:01.059Z', index_id='68767ca6e01b53f51c3f2ac5', video_id='68767ce06c4253f85f0820d0', status='ready', system_metadata={'filename': 'twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4', 'duration': 1448.88, 'width': 640, 'height': 480}, hls=TaskHLS(video_url='', thumbnail_urls=[], status='PROCESSING', updated_at='2025-07-15T16:08:01.059Z'))
Uploaded https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4. The unique identifer of your video is 68767ce06c4253f85f0820d0.
Task created successfully! Task ID: 68767d7a03f1a1f6cd14797d
  Status=pending
  Status=indexing
  Status=ready
Indexing done: Task(id='68767d7a03f1a1f6cd14797d', created_at='2025-07-15T16:10:37.601Z', updated_at='2025-07-15T16:10:37.601Z', index_id='68767ca6e01b53f51c3f2ac5', video_id='68767d7a03f1a1f6cd14797d', status='ready', system_metadata={'filename': 'AnimationTest.mov', 'duration': 24.45679, 'width': 720, 'height': 405}, hls=TaskHLS(video_url='', thumbnail_urls=[], status='PROCESSING', updated_at='2025-07-15T16:10:37.601Z'))
Uploaded https://archive.org/download/The_Worm_in_the_Apple_Animation_Test/AnimationTest.mov. The unique identifer of your video is 68767d7a03f1a1f6cd14797d.
```

Now that the videos have been uploaded, we can generate the keywords, and summaries on the videos below. You will notice on the output that the video uploaded last is the one that is processed first in this stage. This matters since we store other attributes on the videos on arrays (eg URLs, Titles).

In \[13\]:

Copied!

```
import textwrap
client = TwelveLabs(api_key=TL_API_KEY)


summaries = []
keywords_array = []

# Get all videos in an Index
videos = client.index.video.list(index_id)
for video in videos:
    print(f"Generating text for {video.id}")

    res = client.summarize(
        video_id=video.id,
        type="summary",
        prompt="Generate an abstract of the video serving as metadata on the video, up to five sentences.",
    )
    
    wrapped = textwrap.wrap(res.summary, width=110)
    print("Summary:")
    print("\n".join(wrapped))
    summaries.append(res.summary)

    keywords = client.analyze(
        video_id=video.id,
        prompt="Based on this video, I want to generate five keywords for SEO (Search Engine Optimization). Provide just the keywords as a comma delimited list without any additional text.",
    )
    print(f"Open-ended Text: {keywords.data}")
    keywords_array.append(keywords.data)
```

import textwrap client = TwelveLabs(api_key=TL_API_KEY) summaries = [] keywords_array = []

# Get all videos in an Index

videos = client.index.video.list(index_id) for video in videos: print(f"Generating text for {video.id}") res = client.summarize( video_id=video.id, type="summary", prompt="Generate an abstract of the video serving as metadata on the video, up to five sentences.", ) wrapped = textwrap.wrap(res.summary, width=110) print("Summary:") print("\\n".join(wrapped)) summaries.append(res.summary) keywords = client.analyze( video_id=video.id, prompt="Based on this video, I want to generate five keywords for SEO (Search Engine Optimization). Provide just the keywords as a comma delimited list without any additional text.", ) print(f"Open-ended Text: {keywords.data}") keywords_array.append(keywords.data)

```
Generating text for 68767d7a03f1a1f6cd14797d
Summary:
The video titled "The Worm in the Apple Animation Test" showcases a whimsical scene where a segmented worm
emerges from a red apple, positioned on the left side of the frame, and moves across a green field under a
cloudy sky. As the worm progresses, its segments detach one by one, leaving the head connected to the last
segment, with the detached parts scattered around the base of the hill where the apple rests. The camera zooms
out to reveal more of the grassy terrain and then focuses closely on the worm's face, which exhibits a range
of expressions from surprise to anger, enhancing the animated narrative. The worm's journey ends as it crawls
off-screen, leaving behind a visually engaging and animated sequence. The video is accompanied by a
repetitive, light-hearted musical score that adds to the playful tone of the animation.
Open-ended Text: worm, apple, animation, test, victor lyuboslavsky
Generating text for 68767ce06c4253f85f0820d0
Summary:
The video is an animated adaptation of "Twas The Night Before Christmas," featuring a blend of human and mouse
characters. It begins with a snowy night scene and transitions to a clockmaker's workshop, where the
clockmaker, Joshua Trundle, and his family face challenges after a critical letter to Santa is written by
Albert, Trundle's son. The story unfolds with the town's efforts to reconcile with Santa through a special
clock designed to play a welcoming song on Christmas Eve, but complications arise when the clock malfunctions.
Despite the setbacks, the family and community work together to fix the clock and restore belief in Santa,
culminating in his magical arrival, bringing joy and gifts to all. The video concludes with a heartfelt
message about the power of belief and the importance of making amends.
Open-ended Text: snowy village, clock tower, Santa Claus, mechanical gears, Christmas chimes
Generating text for 68767caa47c93cd3ab1e4b05
Summary:
The video captures a serene snowy landscape with pine trees under a cloudy sky, where a squirrel emerges from
behind a rock formation carrying an acorn. Upon noticing another acorn in the foreground, the squirrel appears
momentarily surprised, as indicated by its vocalization "Oh...". It then drops one acorn and begins to nibble
on the other, eventually discarding fragments of it before leaping away. The scene concludes with the
squirrel's departure, leaving behind the remnants of the acorn, as darkness gradually engulfs the snowy
setting.
Open-ended Text: squirrel, acorn, winter, snow, forest
```

We need to store the titles of the videos as an additional attribute.

In \[14\]:

Copied!

```
# Creating array with titles
titles = [
    "The Worm in the Apple Animation Test",
    "Twas the night before Christmas",
    "The END (Blue Sky Studios)",
]
```

# Creating array with titles

titles = [ "The Worm in the Apple Animation Test", "Twas the night before Christmas", "The END (Blue Sky Studios)", ]

## 2.2 Generate Embeddings[¶](#22-generate-embeddings)

The following code leverages the [Embed API](https://docs.twelvelabs.io/docs/create-video-embeddings) to create an asynchronous embedding task to embed the sample videos.

Twelve Labs video embeddings capture all the subtle cues and interactions between different modalities, including the visual expressions, body language, spoken words, and the overall context of the video, encapsulating the essence of all these modalities and their interrelations over time.

In \[15\]:

Copied!

```
client = TwelveLabs(api_key=TL_API_KEY)

# Initialize an array to store the task IDs as strings
task_ids = []

for url in VIDEO_URLs:
    task = client.embed.task.create(model_name="Marengo-retrieval-2.7", video_url=url)
    print(
        f"Created task: id={task.id} model_name={task.model_name} status={task.status}"
    )
    # Append the task ID to the array
    task_ids.append(str(task.id))
    status = task.wait_for_done(sleep_interval=10, callback=on_task_update)
    print(f"Embedding done: {status}")
    if task.status != "ready":
        raise RuntimeError(f"Embedding failed with status {task.status}")
```

client = TwelveLabs(api_key=TL_API_KEY)

# Initialize an array to store the task IDs as strings

task_ids = [] for url in VIDEO_URLs: task = client.embed.task.create(model_name="Marengo-retrieval-2.7", video_url=url) print( f"Created task: id={task.id} model_name={task.model_name} status={task.status}" )

# Append the task ID to the array

task_ids.append(str(task.id)) status = task.wait_for_done(sleep_interval=10, callback=on_task_update) print(f"Embedding done: {status}") if task.status != "ready": raise RuntimeError(f"Embedding failed with status {task.status}")

```
Created task: id=6876856e4fc16ea9b2fdb823 model_name=Marengo-retrieval-2.7 status=processing
  Status=processing
  Status=processing
  Status=ready
Embedding done: ready
Created task: id=68768593de7e2a0235058cc6 model_name=Marengo-retrieval-2.7 status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=processing
  Status=ready
Embedding done: ready
Created task: id=6876860547c93cd3ab1e4cd7 model_name=Marengo-retrieval-2.7 status=processing
  Status=processing
  Status=ready
Embedding done: ready
```

## 2.3 Retrieve Embeddings[¶](#23-retrieve-embeddings)

Once the embedding task is completed, we can retrieve the results of the embedding task based on the task_ids.

In \[16\]:

Copied!

```
# Spin-up session
client = TwelveLabs(api_key=TL_API_KEY)

# Initialize an array to store the task objects directly
tasks = []

for task_id in task_ids:
    # Retrieve the task
    task = client.embed.task.retrieve(task_id)
    tasks.append(task)

    # Print task details
    print(f"Task ID: {task.id}")
    print(f"Status: {task.status}")
```

# Spin-up session

client = TwelveLabs(api_key=TL_API_KEY)

# Initialize an array to store the task objects directly

tasks = [] for task_id in task_ids:

# Retrieve the task

task = client.embed.task.retrieve(task_id) tasks.append(task)

# Print task details

print(f"Task ID: {task.id}") print(f"Status: {task.status}")

```
Task ID: 6876856e4fc16ea9b2fdb823
Status: ready
Task ID: 68768593de7e2a0235058cc6
Status: ready
Task ID: 6876860547c93cd3ab1e4cd7
Status: ready
```

We can now review the output structure of the first segment for each one of these videos. This output will help us define the schema to store the embeddings in Vespa in the second part of this notebook.

From looking at this output, the video has been embedded into chunks of 6 seconds each (default configurable value in the Embed API). Each embedding has a float vector of dimension 1024.

The number of segments generated vary per video, based on the length of the videos ranging from 37 to 242 segments.

In \[17\]:

Copied!

```
for task in tasks:
    print(task.id)
    # Display data types of each field
    for key, value in task.video_embedding.segments[0]:
        if isinstance(value, list):
            print(
                f"{key}: list of size {len(value)} (truncated to 5 items): {value[:5]} "
            )
        else:
            print(f"{key}: {type(value).__name__} : {value}")
    print(f"Total Number of segments: {len(task.video_embedding.segments)}")
```

for task in tasks: print(task.id)

# Display data types of each field

for key, value in task.video_embedding.segments\[0\]: if isinstance(value, list): print( f"{key}: list of size {len(value)} (truncated to 5 items): {value[:5]} " ) else: print(f"{key}: {type(value).__name__} : {value}") print(f"Total Number of segments: {len(task.video_embedding.segments)}")

```
6876856e4fc16ea9b2fdb823
start_offset_sec: float : 0.0
end_offset_sec: float : 6.0
embedding_scope: str : clip
embedding_option: str : visual-text
embeddings_float: list of size 1024 (truncated to 5 items): [0.0227238, -0.002079417, 0.01519275, -0.009030234, -0.00162781] 
Total Number of segments: 12
68768593de7e2a0235058cc6
start_offset_sec: float : 0.0
end_offset_sec: float : 6.0
embedding_scope: str : clip
embedding_option: str : visual-text
embeddings_float: list of size 1024 (truncated to 5 items): [0.024328815, -0.0035867887, 0.016065866, 0.02501548, 0.007778642] 
Total Number of segments: 484
6876860547c93cd3ab1e4cd7
start_offset_sec: float : 0.0
end_offset_sec: float : 6.0
embedding_scope: str : clip
embedding_option: str : visual-text
embeddings_float: list of size 1024 (truncated to 5 items): [0.05419811, -0.0018933096, 0.008044507, -0.01940344, 0.013152712] 
Total Number of segments: 8
```

# 3. Deploy a Vespa Application[¶](#3-deploy-a-vespa-application)

At this point, we are ready to deploy a Vespa Application. We have generated the attributes we needed on each video, as well as the embeddings.

## 3.1 Create an Application Package[¶](#31-create-an-application-package)

The [application package](https://vespa-engine.github.io/pyvespa/api/vespa/package.md) has all the Vespa configuration files - create one from scratch:

The Vespa schema deployed as part of the package is called `videos`. All the fields are matching the output of the Twelvelabs Embed API above. Refer to the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html) for more information on the schema specification.

We can first define the schema using pyvespa

In \[18\]:

Copied!

```
videos_schema = Schema(
    name="videos",
    document=Document(
        fields=[
            Field(name="video_url", type="string", indexing=["summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="keywords",
                type="string",
                indexing=["index", "summary"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="video_summary",
                type="string",
                indexing=["index", "summary"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="embedding_scope", type="string", indexing=["attribute", "summary"]
            ),
            Field(
                name="start_offset_sec",
                type="array<float>",
                indexing=["attribute", "summary"],
            ),
            Field(
                name="end_offset_sec",
                type="array<float>",
                indexing=["attribute", "summary"],
            ),
            Field(
                name="embeddings",
                type="tensor<float>(p{},x[1024])",
                indexing=["index", "attribute"],
                ann=HNSW(distance_metric="angular"),
            ),
        ]
    ),
)

fieldsets = (
    [
        FieldSet(
            name="default",
            fields=["title", "keywords", "video_summary"],
        ),
    ],
)

mapfunctions = [
    Function(
        name="similarities",
        expression="""
                      sum(
                          query(q) * attribute(embeddings), x
                          )
                      """,
    ),
    Function(
        name="bm25_score",
        expression="bm25(title) + bm25(keywords) + bm25(video_summary)",
    ),
]

semantic_rankprofile = RankProfile(
    name="hybrid",
    inputs=[("query(q)", "tensor<float>(x[1024])")],
    first_phase="bm25_score",
    second_phase=SecondPhaseRanking(
        expression="closeness(field, embeddings)", rerank_count=10
    ),
    match_features=["closest(embeddings)"],
    summary_features=["similarities"],
    functions=mapfunctions,
)

videos_schema.add_rank_profile(semantic_rankprofile)
```

videos_schema = Schema( name="videos", document=Document( fields=\[ Field(name="video_url", type="string", indexing=["summary"]), Field( name="title", type="string", indexing=["index", "summary"], match=["text"], index="enable-bm25", ), Field( name="keywords", type="string", indexing=["index", "summary"], match=["text"], index="enable-bm25", ), Field( name="video_summary", type="string", indexing=["index", "summary"], match=["text"], index="enable-bm25", ), Field( name="embedding_scope", type="string", indexing=["attribute", "summary"] ), Field( name="start_offset_sec", type="array<float>", indexing=["attribute", "summary"], ), Field( name="end_offset_sec", type="array<float>", indexing=["attribute", "summary"], ), Field( name="embeddings", type="tensor<float>(p{},x[1024])", indexing=["index", "attribute"], ann=HNSW(distance_metric="angular"), ), \] ), ) fieldsets = ( \[ FieldSet( name="default", fields=["title", "keywords", "video_summary"], ), \], ) mapfunctions = [ Function( name="similarities", expression=""" sum( query(q) * attribute(embeddings), x ) """, ), Function( name="bm25_score", expression="bm25(title) + bm25(keywords) + bm25(video_summary)", ), ] semantic_rankprofile = RankProfile( name="hybrid", inputs=\[("query(q)", "tensor<float>(x[1024])")\], first_phase="bm25_score", second_phase=SecondPhaseRanking( expression="closeness(field, embeddings)", rerank_count=10 ), match_features=["closest(embeddings)"], summary_features=["similarities"], functions=mapfunctions, ) videos_schema.add_rank_profile(semantic_rankprofile)

We can now create the package based on the previous schema

In \[19\]:

Copied!

```
# Create the Vespa application package
package = ApplicationPackage(name=application, schema=[videos_schema])
```

# Create the Vespa application package

package = ApplicationPackage(name=application, schema=[videos_schema])

## 3.2 Deploy the Application Package[¶](#32-deploy-the-application-package)

The app is now defined and ready to deploy to Vespa Cloud.

Deploy `package` to Vespa Cloud, by creating an instance of [VespaCloud](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#VespaCloud):

In \[20\]:

Copied!

```
vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    application_package=package,
    key_content=os.getenv("VESPA_TEAM_API_KEY", None),
)
```

vespa_cloud = VespaCloud( tenant=tenant_name, application=application, application_package=package, key_content=os.getenv("VESPA_TEAM_API_KEY", None), )

```
Setting application...
Running: vespa config set application vespa-presales.videosearch.default
Setting target cloud...
Running: vespa config set target cloud

No api-key found for control plane access. Using access token.
Checking for access token in auth.json...
Access token expired. Please re-authenticate.
Your Device Confirmation code is: MJKL-VTBW
Automatically open confirmation page in your default browser? [Y/n] 
Opened link in your browser: https://login.console.vespa-cloud.com/activate?user_code=MJKL-VTBW
Waiting for login to complete in browser ... done;1m⣽
Success: Logged in
 auth.json created at /Users/zohar/.vespa/auth.json
Successfully obtained access token for control plane access.
```

In \[21\]:

Copied!

```
app = vespa_cloud.deploy()
```

app = vespa_cloud.deploy()

```
Deployment started in run 19 of dev-aws-us-east-1c for vespa-presales.videosearch. This may take a few minutes the first time.
INFO    [16:48:18]  Deploying platform version 8.547.15 and application dev build 11 for dev-aws-us-east-1c of default ...
INFO    [16:48:18]  Using CA signed certificate version 3
INFO    [16:48:18]  Using 1 nodes in container cluster 'videosearch_container'
INFO    [16:48:21]  Session 7523 for tenant 'vespa-presales' prepared and activated.
INFO    [16:48:21]  ######## Details for all nodes ########
INFO    [16:48:21]  h121570a.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [16:48:21]  --- platform vespa/cloud-tenant-rhel8:8.547.15
INFO    [16:48:21]  --- container on port 4080 has config generation 7522, wanted is 7523
INFO    [16:48:21]  --- metricsproxy-container on port 19092 has config generation 7522, wanted is 7523
INFO    [16:48:21]  h119160h.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [16:48:21]  --- platform vespa/cloud-tenant-rhel8:8.547.15
INFO    [16:48:21]  --- container-clustercontroller on port 19050 has config generation 7522, wanted is 7523
INFO    [16:48:21]  --- metricsproxy-container on port 19092 has config generation 7523, wanted is 7523
INFO    [16:48:21]  h117409h.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [16:48:21]  --- platform vespa/cloud-tenant-rhel8:8.547.15
INFO    [16:48:21]  --- logserver-container on port 4080 has config generation 7523, wanted is 7523
INFO    [16:48:21]  --- metricsproxy-container on port 19092 has config generation 7522, wanted is 7523
INFO    [16:48:21]  h121486b.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [16:48:21]  --- platform vespa/cloud-tenant-rhel8:8.547.15
INFO    [16:48:21]  --- storagenode on port 19102 has config generation 7522, wanted is 7523
INFO    [16:48:21]  --- searchnode on port 19107 has config generation 7523, wanted is 7523
INFO    [16:48:21]  --- distributor on port 19111 has config generation 7523, wanted is 7523
INFO    [16:48:21]  --- metricsproxy-container on port 19092 has config generation 7523, wanted is 7523
INFO    [16:48:29]  Found endpoints:
INFO    [16:48:29]  - dev.aws-us-east-1c
INFO    [16:48:29]   |-- https://d4ed0f5e.ee8b6819.z.vespa-app.cloud/ (cluster 'videosearch_container')
INFO    [16:48:30]  Deployment of new application revision complete!
Only region: aws-us-east-1c available in dev environment.
Found mtls endpoint for videosearch_container
URL: https://d4ed0f5e.ee8b6819.z.vespa-app.cloud/
Application is up!
```

## 3.3 Feed the Vespa Application[¶](#33-feed-the-vespa-application)

The `vespa_feed` feed format for `pyvespa` expects a dict with the keys `id` and `fields`:

`{ "id": "vespa-document-id", "fields": {"vespa_field": "vespa-field-value"}}`

For the id, we will use a md5 hash of the video url.

The video embedding output segments are added to the `fields` in `vespa_feed`.

In \[22\]:

Copied!

```
# Initialize a list to store Vespa feed documents
vespa_feed = []

# Need to reverse VIDEO_URLS as keywords/summaries generated in reverse order
VIDEO_URLs.reverse()

# Iterate through each task and corresponding metadata
for i, task in enumerate(tasks):
    video_url = VIDEO_URLs[i]
    title = titles[i]
    keywords = keywords_array[i]
    summary = summaries[i]

    start_offsets = []  # Reset for each video
    end_offsets = []  # Reset for each video
    embeddings = {}  # Reset for each video

    # Iterate through the video embedding segments
    for index, segment in enumerate(task.video_embedding.segments):
        # Append start and end offsets as floats
        start_offsets.append(float(segment.start_offset_sec))
        end_offsets.append(float(segment.end_offset_sec))

        # Add embedding to a multi-dimensional dictionary with index as the key
        embeddings[str(index)] = list(map(float, segment.embeddings_float))

    # Create Vespa document for each task
    for segment in task.video_embedding.segments:
        start_offset_sec = segment.start_offset_sec
        end_offset_sec = segment.end_offset_sec
        embedding = list(map(float, segment.embeddings_float))

        # Create a unique ID by hashing the URL and segment index
        id_hash = hashlib.md5(f"{video_url}_{index}".encode()).hexdigest()

        document = {
            "id": id_hash,
            "fields": {
                "video_url": video_url,
                "title": title,
                "keywords": keywords,
                "video_summary": summary,
                "embedding_scope": segment.embedding_scope,
                "start_offset_sec": start_offsets,
                "end_offset_sec": end_offsets,
                "embeddings": embeddings,
            },
        }
    vespa_feed.append(document)
```

# Initialize a list to store Vespa feed documents

vespa_feed = []

# Need to reverse VIDEO_URLS as keywords/summaries generated in reverse order

VIDEO_URLs.reverse()

# Iterate through each task and corresponding metadata

for i, task in enumerate(tasks): video_url = VIDEO_URLs[i] title = titles[i] keywords = keywords_array[i] summary = summaries[i] start_offsets = [] # Reset for each video end_offsets = [] # Reset for each video embeddings = {} # Reset for each video

# Iterate through the video embedding segments

for index, segment in enumerate(task.video_embedding.segments):

# Append start and end offsets as floats

start_offsets.append(float(segment.start_offset_sec)) end_offsets.append(float(segment.end_offset_sec))

# Add embedding to a multi-dimensional dictionary with index as the key

embeddings[str(index)] = list(map(float, segment.embeddings_float))

# Create Vespa document for each task

for segment in task.video_embedding.segments: start_offset_sec = segment.start_offset_sec end_offset_sec = segment.end_offset_sec embedding = list(map(float, segment.embeddings_float))

# Create a unique ID by hashing the URL and segment index

id_hash = hashlib.md5(f"{video_url}\_{index}".encode()).hexdigest() document = { "id": id_hash, "fields": { "video_url": video_url, "title": title, "keywords": keywords, "video_summary": summary, "embedding_scope": segment.embedding_scope, "start_offset_sec": start_offsets, "end_offset_sec": end_offsets, "embeddings": embeddings, }, } vespa_feed.append(document)

We can quickly validate the number of the number of documents created (one for each video), and visually check the first record.

In \[23\]:

Copied!

```
# Print Vespa feed size and an example
print(f"Total documents created: {len(vespa_feed)}")
```

# Print Vespa feed size and an example

print(f"Total documents created: {len(vespa_feed)}")

```
Total documents created: 3
```

In \[24\]:

Copied!

```
# The positional index of the document
i = 0

# Iterate through the first 3 embeddings in vespa_feed
for i in range(
    min(3, len(vespa_feed))
):  # Ensure we don't exceed the length of vespa_feed
    # Limit the embedding to the first 3 keys and first 5 values for each key
    embedding = vespa_feed[i]["fields"]["embeddings"]
    embedding_sample = {key: values[:3] for key, values in list(embedding.items())[:3]}

# Beautify and print the first document with only the first 5 embedding values
pretty_json = json.dumps(
    {
        "id": vespa_feed[i]["id"],
        "fields": {
            "video_url": vespa_feed[i]["fields"]["video_url"],
            "title": vespa_feed[i]["fields"]["title"],
            "keywords": vespa_feed[i]["fields"]["keywords"],
            "video_summary": vespa_feed[i]["fields"]["video_summary"],
            "embedding_scope": vespa_feed[i]["fields"]["embedding_scope"],
            "start_offset_sec": vespa_feed[i]["fields"]["start_offset_sec"][:3],
            "end_offset_sec": vespa_feed[i]["fields"]["end_offset_sec"][:3],
            "embedding": embedding_sample,
        },
    },
    indent=4,
)

print(pretty_json)
```

# The positional index of the document

i = 0

# Iterate through the first 3 embeddings in vespa_feed

for i in range( min(3, len(vespa_feed)) ): # Ensure we don't exceed the length of vespa_feed

# Limit the embedding to the first 3 keys and first 5 values for each key

embedding = vespa_feed[i]["fields"]["embeddings"] embedding_sample = {key: values[:3] for key, values in list(embedding.items())[:3]}

# Beautify and print the first document with only the first 5 embedding values

pretty_json = json.dumps( { "id": vespa_feed[i]["id"], "fields": { "video_url": vespa_feed[i]["fields"]["video_url"], "title": vespa_feed[i]["fields"]["title"], "keywords": vespa_feed[i]["fields"]["keywords"], "video_summary": vespa_feed[i]["fields"]["video_summary"], "embedding_scope": vespa_feed[i]["fields"]["embedding_scope"], "start_offset_sec": vespa_feed[i]["fields"]["start_offset_sec"][:3], "end_offset_sec": vespa_feed[i]["fields"]["end_offset_sec"][:3], "embedding": embedding_sample, }, }, indent=4, ) print(pretty_json)

```
{
    "id": "93d8476bee530eb39a2122f586d0d13a",
    "fields": {
        "video_url": "https://archive.org/download/the-end-blue-sky-studios/The%20End%281080P_60FPS%29.ia.mp4",
        "title": "The END (Blue Sky Studios)",
        "keywords": "squirrel, acorn, winter, snow, forest",
        "video_summary": "The video captures a serene snowy landscape with pine trees under a cloudy sky, where a squirrel emerges from behind a rock formation carrying an acorn. Upon noticing another acorn in the foreground, the squirrel appears momentarily surprised, as indicated by its vocalization \"Oh...\". It then drops one acorn and begins to nibble on the other, eventually discarding fragments of it before leaping away. The scene concludes with the squirrel's departure, leaving behind the remnants of the acorn, as darkness gradually engulfs the snowy setting.",
        "embedding_scope": "clip",
        "start_offset_sec": [
            0.0,
            6.0,
            12.0
        ],
        "end_offset_sec": [
            6.0,
            12.0,
            18.0
        ],
        "embedding": {
            "0": [
                0.05419811,
                -0.0018933096,
                0.008044507
            ],
            "1": [
                0.016035125,
                -0.015930071,
                0.022429857
            ],
            "2": [
                0.014023403,
                -0.012773005,
                0.019988379
            ]
        }
    }
}
```

Now we can feed to Vespa using `feed_iterable` which accepts any `Iterable` and an optional callback function where we can check the outcome of each operation.

In \[25\]:

Copied!

```
def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )


# Feed data into Vespa synchronously
app.feed_iterable(vespa_feed, schema="videos", callback=callback)
```

def callback(response: VespaResponse, id: str): if not response.is_successful(): print( f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}" )

# Feed data into Vespa synchronously

app.feed_iterable(vespa_feed, schema="videos", callback=callback)

# 4. Performing search on the videos[¶](#4-performing-search-on-the-videos)

## 4.1 Performing a hybrid search on the video[¶](#41-performing-a-hybrid-search-on-the-video)

As an example query, we will retrieve all the chunks which shows Santa Claus on his sleigh. The first step is to generate a text embedding for `Santa Claus on his sleigh` using the `Marengo-retrieval-2.7` model.

In \[28\]:

Copied!

```
client = TwelveLabs(api_key=TL_API_KEY)
user_query = "Santa Claus on his sleigh"

res = client.embed.create(
    model_name="Marengo-retrieval-2.7",
    text=user_query,
)

print("Created a text embedding")
print(f" Model: {res.model_name}")
if res.text_embedding is not None and res.text_embedding.segments is not None:
    q_embedding = res.text_embedding.segments[0].embeddings_float
    print(f" Embedding Dimension: {len(q_embedding)}")
    print(f" Sample 5 values from array: {q_embedding[:5]}")
```

client = TwelveLabs(api_key=TL_API_KEY) user_query = "Santa Claus on his sleigh" res = client.embed.create( model_name="Marengo-retrieval-2.7", text=user_query, ) print("Created a text embedding") print(f" Model: {res.model_name}") if res.text_embedding is not None and res.text_embedding.segments is not None: q_embedding = res.text_embedding.segments[0].embeddings_float print(f" Embedding Dimension: {len(q_embedding)}") print(f" Sample 5 values from array: {q_embedding[:5]}")

```
Created a text embedding
 Model: Marengo-retrieval-2.7
 Embedding Dimension: 1024
 Sample 5 values from array: [-0.018066406, -0.0065307617, 0.05859375, -0.033447266, -0.02368164]
```

The following uses dense vector representations of the query embedding obtained previously and document and matching is performed and accelerated by Vespa's support for [approximate nearest neighbor search](https://docs.vespa.ai/en/approximate-nn-hnsw.html).

The output is limited to the top 1 hit, as we only have a sample of 3 videos. The top hit returned was based on a hybrid ranking based on a bm25 ranking based on a lexical search on the text, keywords and summary of the video, performed as a first phase, and similarity search on the embeddings.

We can see as part of the `match-features`, the segment 212 in the video was the one providing the highest match.

We also calculate the similarities as part of the `summary-features` for the rest of the segments so we can look for top N segments within a video, optionally.

In \[29\]:

Copied!

```
with app.syncio(connections=1) as session:
    response: VespaQueryResponse = session.query(
        yql="select * from videos where userQuery() OR ({targetHits:100}nearestNeighbor(embeddings,q))",
        query=user_query,
        ranking="hybrid",
        hits=1,
        body={"input.query(q)": q_embedding},
    )
    assert response.is_successful()

hit = response.hits[0]

# Extract metadata
doc_id = hit.get("id")
relevance = hit.get("relevance")
source = hit.get("source")
fields = hit.get("fields", {})

# Extract the embedding match cell index (first key in matchfeatures)
match_cells = fields.get("matchfeatures", {}).get("closest(embeddings)", {}).get("cells", {})
if not match_cells:
    raise ValueError("No cells found in matchfeatures.closest(embeddings)")

# Get the first (and only) cell key and value
cell_index, cell_value = next(iter(match_cells.items()))
cell_index = int(cell_index)  # Convert key from string to int

# Extract aligned fields using the index
start_offset = fields.get("start_offset_sec", [])[cell_index]
end_offset = fields.get("end_offset_sec", [])[cell_index]
similarity = fields.get("summaryfeatures", {}).get("similarities", {}).get("cells", {}).get(str(cell_index))

# Print full info
print("Document Metadata:")
print(f"documentid: {doc_id}")
print(f"Relevance: {relevance}")
print(f"Source: {source}")
print(f"Match Features: {fields.get('matchfeatures', 'N/A')}")
print()

print(f"Title: {fields.get('title', 'N/A')}")
print(f"Keywords: {fields.get('keywords', 'N/A')}")
print(f"Video URL: {fields.get('video_url', 'N/A')}")
print(f"Video Summary: {fields.get('video_summary', 'N/A')}")
print(f"Embedding Scope: {fields.get('embedding_scope', 'N/A')}")
print()

# Print details for the matched cell
print(f"Details for cell {cell_index}:")
print(f"Start offset: {start_offset} sec")
print(f"End offset: {end_offset} sec")
print(f"Similarity score: {similarity}")
print(f"Match feature score: {cell_value}")
```

with app.syncio(connections=1) as session: response: VespaQueryResponse = session.query( yql="select * from videos where userQuery() OR ({targetHits:100}nearestNeighbor(embeddings,q))", query=user_query, ranking="hybrid", hits=1, body={"input.query(q)": q_embedding}, ) assert response.is_successful() hit = response.hits[0]

# Extract metadata

doc_id = hit.get("id") relevance = hit.get("relevance") source = hit.get("source") fields = hit.get("fields", {})

# Extract the embedding match cell index (first key in matchfeatures)

match_cells = fields.get("matchfeatures", {}).get("closest(embeddings)", {}).get("cells", {}) if not match_cells: raise ValueError("No cells found in matchfeatures.closest(embeddings)")

# Get the first (and only) cell key and value

cell_index, cell_value = next(iter(match_cells.items())) cell_index = int(cell_index) # Convert key from string to int

# Extract aligned fields using the index

start_offset = fields.get("start_offset_sec", [])[cell_index] end_offset = fields.get("end_offset_sec", [])[cell_index] similarity = fields.get("summaryfeatures", {}).get("similarities", {}).get("cells", {}).get(str(cell_index))

# Print full info

print("Document Metadata:") print(f"documentid: {doc_id}") print(f"Relevance: {relevance}") print(f"Source: {source}") print(f"Match Features: {fields.get('matchfeatures', 'N/A')}") print() print(f"Title: {fields.get('title', 'N/A')}") print(f"Keywords: {fields.get('keywords', 'N/A')}") print(f"Video URL: {fields.get('video_url', 'N/A')}") print(f"Video Summary: {fields.get('video_summary', 'N/A')}") print(f"Embedding Scope: {fields.get('embedding_scope', 'N/A')}") print()

# Print details for the matched cell

print(f"Details for cell {cell_index}:") print(f"Start offset: {start_offset} sec") print(f"End offset: {end_offset} sec") print(f"Similarity score: {similarity}") print(f"Match feature score: {cell_value}")

```
Document Metadata:
documentid: id:videos:videos::d4175516790d7e55a79eb7f190495a92
Relevance: 0.47162757625475055
Source: videosearch_content
Match Features: {'closest(embeddings)': {'type': 'tensor<float>(p{})', 'cells': {'212': 1.0}}}

Title: Twas the night before Christmas
Keywords: snowy village, clock tower, Santa Claus, mechanical gears, Christmas chimes
Video URL: https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4
Video Summary: The video is an animated adaptation of "Twas The Night Before Christmas," featuring a blend of human and mouse characters. It begins with a snowy night scene and transitions to a clockmaker's workshop, where the clockmaker, Joshua Trundle, and his family face challenges after a critical letter to Santa is written by Albert, Trundle's son. The story unfolds with the town's efforts to reconcile with Santa through a special clock designed to play a welcoming song on Christmas Eve, but complications arise when the clock malfunctions. Despite the setbacks, the family and community work together to fix the clock and restore belief in Santa, culminating in his magical arrival, bringing joy and gifts to all. The video concludes with a heartfelt message about the power of belief and the importance of making amends.
Embedding Scope: clip

Details for cell 212:
Start offset: 1272.0 sec
End offset: 1278.0 sec
Similarity score: 0.43537065386772156
Match feature score: 1.0
```

You should see output similar to this:

````
Document
documentid: id:videos:videos::d4175516790d7e55a79eb7f190495a92
Relevance: 0.47162757625475055
Source: videosearch_content
Match Features: {'closest(embeddings)': {'type': 'tensor<float>(p{})', 'cells': {'212': 1.0}}}

Title: Twas the night before Christmas
Keywords: snowy village, clock tower, Santa Claus, mechanical gears, Christmas chimes
Video URL: https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4
Video Summary: The video is an animated adaptation of "Twas The Night Before Christmas," featuring a blend of human and mouse characters. It begins with a snowy night scene and transitions to a clockmaker's workshop, where the clockmaker, Joshua Trundle, and his family face challenges after a critical letter to Santa is written by Albert, Trundle's son. The story unfolds with the town's efforts to reconcile with Santa through a special clock designed to play a welcoming song on Christmas Eve, but complications arise when the clock malfunctions. Despite the setbacks, the family and community work together to fix the clock and restore belief in Santa, culminating in his magical arrival, bringing joy and gifts to all. The video concludes with a heartfelt message about the power of belief and the importance of making amends.
Embedding Scope: clip

Details for cell 212:
Start offset: 1272.0 sec
End offset: 1278.0 sec
Similarity score: 0.43537065386772156
Match feature score: 1.0```
````

In order to process the results above in a more consumable format and sort out the top N segments based on similarities, we can do this more conveniently in a pandas dataframe below:

In \[37\]:

Copied!

```
def get_top_n_similarity_matches(data, N=5):
    """
    Function to extract the top N similarity scores and their corresponding start and end offsets.

    Args:
    - data (dict): Input JSON-like structure containing similarities and offsets.
    - N (int): The number of top similarity scores to return.

    Returns:
    - pd.DataFrame: A DataFrame with the top N similarity scores and their corresponding offsets.
    """
    # Extract relevant fields
    similarities = data["fields"]["summaryfeatures"]["similarities"]["cells"]
    start_offset_sec = data["fields"]["start_offset_sec"]
    end_offset_sec = data["fields"]["end_offset_sec"]

    # Convert similarity scores to a list of tuples (index, similarity_score) and sort by similarity score
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Extract top N similarity scores
    top_n_similarities = sorted_similarities[:N]

    # Prepare results
    results = []
    for index_str, score in top_n_similarities:
        index = int(index_str)
        if index < len(start_offset_sec):
            result = {
                "index": index,
                "similarity_score": score,
                "start_offset_sec": start_offset_sec[index],
                "end_offset_sec": end_offset_sec[index],
            }
        else:
            result = {
                "index": index,
                "similarity_score": score,
                "start_offset_sec": None,
                "end_offset_sec": None,
            }
        results.append(result)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    return df
```

def get_top_n_similarity_matches(data, N=5): """ Function to extract the top N similarity scores and their corresponding start and end offsets. Args:

- data (dict): Input JSON-like structure containing similarities and offsets.
- N (int): The number of top similarity scores to return. Returns:
- pd.DataFrame: A DataFrame with the top N similarity scores and their corresponding offsets. """

# Extract relevant fields

similarities = data["fields"]["summaryfeatures"]["similarities"]["cells"] start_offset_sec = data["fields"]["start_offset_sec"] end_offset_sec = data["fields"]["end_offset_sec"]

# Convert similarity scores to a list of tuples (index, similarity_score) and sort by similarity score

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# Extract top N similarity scores

top_n_similarities = sorted_similarities[:N]

# Prepare results

results = [] for index_str, score in top_n_similarities: index = int(index_str) if index < len(start_offset_sec): result = { "index": index, "similarity_score": score, "start_offset_sec": start_offset_sec[index], "end_offset_sec": end_offset_sec[index], } else: result = { "index": index, "similarity_score": score, "start_offset_sec": None, "end_offset_sec": None, } results.append(result)

# Convert results to a DataFrame

df = pd.DataFrame(results) return df

In \[38\]:

Copied!

```
df_result = get_top_n_similarity_matches(response.hits[0], N=10)
df_result
```

df_result = get_top_n_similarity_matches(response.hits[0], N=10) df_result

Out\[38\]:

|     | index | similarity_score | start_offset_sec | end_offset_sec |
| --- | ----- | ---------------- | ---------------- | -------------- |
| 0   | 212   | 0.435371         | 1272.0           | 1278.0         |
| 1   | 230   | 0.418007         | 1380.0           | 1386.0         |
| 2   | 210   | 0.411242         | 1260.0           | 1266.0         |
| 3   | 211   | 0.409344         | 1266.0           | 1272.0         |
| 4   | 208   | 0.408644         | 1248.0           | 1254.0         |
| 5   | 231   | 0.406000         | 1386.0           | 1392.0         |
| 6   | 209   | 0.404767         | 1254.0           | 1260.0         |
| 7   | 229   | 0.403729         | 1374.0           | 1380.0         |
| 8   | 203   | 0.403292         | 1218.0           | 1224.0         |
| 9   | 207   | 0.391671         | 1242.0           | 1248.0         |

## 5. Review results (Optional)[¶](#5-review-results-optional)

We can review the results by spinning up a video player in the notebook and check the segments identified and judge by ourselves.

But, first we need to obtain the contiguous segments, add 3 seconds overlap in the consolidated segments and convert to MM:SS so we can quickly find the segments to watch in the player. Let's write a function that takes the response as an input and provides the consolidated segments to view in the player.

In \[40\]:

Copied!

```
def concatenate_contiguous_segments(df):
    """
    Function to concatenate contiguous segments based on their start and end offsets.
    Converts the concatenated segments to MM:SS format.

    Args:
    - df (pd.DataFrame): DataFrame with columns 'start_offset_sec' and 'end_offset_sec'.

    Returns:
    - List of tuples with concatenated segments in MM:SS format as (start_time, end_time).
    """
    if df.empty:
        return []

    # Sort by start_offset_sec for ordered processing
    df = df.sort_values(by="start_offset_sec").reset_index(drop=True)

    # Initialize the list to hold concatenated segments
    concatenated_segments = []

    # Initialize the first segment
    start = df.iloc[0]["start_offset_sec"]
    end = df.iloc[0]["end_offset_sec"]

    for i in range(1, len(df)):
        current_start = df.iloc[i]["start_offset_sec"]
        current_end = df.iloc[i]["end_offset_sec"]

        # Check if the current segment is contiguous with the previous one
        if current_start <= end:
            # Extend the segment if it is contiguous
            end = max(end, current_end)
        else:
            # Add the previous segment to the result list in MM:SS format
            concatenated_segments.append(
                (convert_seconds_to_mmss(start - 3), convert_seconds_to_mmss(end + 3))
            )
            # Start a new segment
            start = current_start
            end = current_end

    # Add the final segment
    concatenated_segments.append(
        (convert_seconds_to_mmss(start - 3), convert_seconds_to_mmss(end + 3))
    )

    return concatenated_segments


def convert_seconds_to_mmss(seconds):
    """
    Converts seconds to MM:SS format.

    Args:
    - seconds (float): Time in seconds.

    Returns:
    - str: Time in MM:SS format.
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"
```

def concatenate_contiguous_segments(df): """ Function to concatenate contiguous segments based on their start and end offsets. Converts the concatenated segments to MM:SS format. Args:

- df (pd.DataFrame): DataFrame with columns 'start_offset_sec' and 'end_offset_sec'. Returns:
- List of tuples with concatenated segments in MM:SS format as (start_time, end_time). """ if df.empty: return []

# Sort by start_offset_sec for ordered processing

df = df.sort_values(by="start_offset_sec").reset_index(drop=True)

# Initialize the list to hold concatenated segments

concatenated_segments = []

# Initialize the first segment

start = df.iloc[0]["start_offset_sec"] end = df.iloc[0]["end_offset_sec"] for i in range(1, len(df)): current_start = df.iloc[i]["start_offset_sec"] current_end = df.iloc[i]["end_offset_sec"]

# Check if the current segment is contiguous with the previous one

if current_start \<= end:

# Extend the segment if it is contiguous

end = max(end, current_end) else:

# Add the previous segment to the result list in MM:SS format

concatenated_segments.append( (convert_seconds_to_mmss(start - 3), convert_seconds_to_mmss(end + 3)) )

# Start a new segment

start = current_start end = current_end

# Add the final segment

concatenated_segments.append( (convert_seconds_to_mmss(start - 3), convert_seconds_to_mmss(end + 3)) ) return concatenated_segments def convert_seconds_to_mmss(seconds): """ Converts seconds to MM:SS format. Args:

- seconds (float): Time in seconds. Returns:
- str: Time in MM:SS format. """ minutes = int(seconds // 60) seconds = int(seconds % 60) return f"{minutes:02}:{seconds:02}"

In \[41\]:

Copied!

```
segments = concatenate_contiguous_segments(df_result)
segments
```

segments = concatenate_contiguous_segments(df_result) segments

Out\[41\]:

```
[('20:15', '20:27'), ('20:39', '21:21'), ('22:51', '23:15')]
```

We can now spin-up the player and review the segments of interest. Video player is set to start in the middle of the first segment.

In \[42\]:

Copied!

```
from IPython.display import HTML

video_url = "https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4"

video_player = f"""
<video id="myVideo" width="640" height="480" controls>
  <source src="{video_url}" type="video/mp4">
  Your browser does not support the video tag.
</video>

"""

HTML(video_player)
```

from IPython.display import HTML video_url = "https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4" video_player = f""" <video id="myVideo" width="640" height="480" controls>

<source src="{video_url}" type="video/mp4">
Your browser does not support the video tag.
</video>
"""
HTML(video_player)

Out\[42\]:

\[

Your browser does not support the video tag. \](https://ia601401.us.archive.org/1/items/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net/twas-the-night-before-christmas-1974-full-movie-freedownloadvideo.net.mp4)

## 6. Clean-up[¶](#6-clean-up)

The following will delete the application and data from the dev environment.

In \[35\]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()

```
Deactivated vespa-presales.videosearch in dev.aws-us-east-1c
Deleted instance vespa-presales.videosearch.default
```

The following will delete the index created earlier where videos where uploaded:

In \[36\]:

Copied!

```
# Creating a client
client = TwelveLabs(api_key=TL_API_KEY)

client.index.delete(index_id)
```

# Creating a client

client = TwelveLabs(api_key=TL_API_KEY) client.index.delete(index_id)
