import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import re
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
import requests
from pathlib import Path
import jieba
from collections import defaultdict
import numpy as np
import csv
from datetime import datetime
import warnings
from api_client import DeepSeekAPIClient
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import CrossEncoder
    import torch
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False