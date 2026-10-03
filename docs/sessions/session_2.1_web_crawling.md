# Session 2.1: Web Crawling with Selenium

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how the crawler dispatcher works
- Be able to create custom web crawlers
- Master Selenium WebDriver setup
- Implement error handling and retries

---

## 🏗️ Architecture Overview

### Crawler System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Crawler Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Input (URLs)                                          │
│       ↓                                                      │
│  CrawlerDispatcher (Routes to correct crawler)              │
│       ↓                                                      │
│  ┌─────────────┬─────────────┬─────────────┐                │
│  │ GitHub      │ Medium      │ LinkedIn    │                │
│  │ Crawler     │ Crawler     │ Crawler     │                │
│  └─────────────┴─────────────┴─────────────┘                │
│       ↓                                                      │
│  BaseCrawler (Common Selenium logic)                        │
│       ↓                                                      │
│  Document Objects (Saved to MongoDB)                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Key Files

### 1. `llm_engineering/application/crawlers/base.py`

**Purpose**: Base class with common Selenium functionality.

```python
# llm_engineering/application/crawlers/base.py

from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller

class BaseCrawler(ABC):
    def __init__(self):
        # Auto-install ChromeDriver
        chromedriver_autoinstaller.install()
        
        # Configure Chrome options
        options = Options()
        options.add_argument("--headless")  # No GUI
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Create WebDriver
        self.driver = webdriver.Chrome(
            service=Service(),
            options=options
        )
    
    def extract(self, link: str, **kwargs) -> list:
        """Main extraction method"""
        try:
            self.driver.get(link)
            return self.extract_content(**kwargs)
        except Exception as e:
            logger.error(f"Error extracting {link}: {e}")
            return []
        finally:
            self.driver.quit()
    
    @abstractmethod
    def extract_content(self, **kwargs) -> list:
        """Implement in subclass"""
        pass
```

**Key Concepts**:
- **Abstract Base Class**: Forces subclasses to implement `extract_content()`
- **Headless Browser**: Runs without GUI (important for servers)
- **Auto-Driver Management**: No manual ChromeDriver installation

---

### 2. `llm_engineering/application/crawlers/dispatcher.py`

**Purpose**: Route URLs to appropriate crawlers.

```python
# llm_engineering/application/crawlers/dispatcher.py

from urllib.parse import urlparse

class CrawlerDispatcher:
    def __init__(self):
        self._crawlers = {}
    
    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        """Factory method to build pre-configured dispatcher"""
        dispatcher = cls()
        
        # Register crawlers for each domain
        dispatcher.register_linkedin()
        dispatcher.register_medium()
        dispatcher.register_github()
        
        return dispatcher
    
    def register(self, domain: str, crawler_class):
        """Register a crawler for a domain"""
        self._crawlers[domain] = crawler_class
    
    def register_linkedin(self):
        from .linkedin import LinkedInCrawler
        self._crawlers["linkedin.com"] = LinkedInCrawler
    
    def register_medium(self):
        from .medium import MediumCrawler
        self._crawlers["medium.com"] = MediumCrawler
        self._crawlers["towardsdatascience.com"] = MediumCrawler
    
    def register_github(self):
        from .github import GitHubCrawler
        self._crawlers["github.com"] = GitHubCrawler
    
    def get_crawler(self, url: str):
        """Get appropriate crawler for URL"""
        domain = urlparse(url).netloc
        
        # Try exact match first
        if domain in self._crawlers:
            return self._crawlers[domain]()
        
        # Try base domain (e.g., www.linkedin.com -> linkedin.com)
        base_domain = domain.replace("www.", "")
        if base_domain in self._crawlers:
            return self._crawlers[base_domain]()
        
        # Default crawler
        from .custom_article import CustomArticleCrawler
        return CustomArticleCrawler()
```

**Usage**:
```python
dispatcher = CrawlerDispatcher.build()
crawler = dispatcher.get_crawler("https://medium.com/article")
documents = crawler.extract(link="https://medium.com/article", user=user_doc)
```

---

### 3. `llm_engineering/application/crawlers/medium.py`

**Purpose**: Extract articles from Medium.

```python
# llm_engineering/application/crawlers/medium.py

from bs4 import BeautifulSoup
from llm_engineering.domain.documents import ArticleDocument

class MediumCrawler(BaseCrawler):
    def extract_content(self, user, **kwargs) -> list:
        """Extract Medium article content"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # Extract title
        title = soup.find('h1', class_='pw-post-title')
        title = title.text.strip() if title else "Unknown"
        
        # Extract subtitle
        subtitle = soup.find('h2', class_='pw-subtitle-paragraph')
        subtitle = subtitle.text.strip() if subtitle else ""
        
        # Extract content
        content_divs = soup.find_all('p')
        content = '\n'.join([p.text.strip() for p in content_divs])
        
        # Create document
        doc = ArticleDocument(
            content=f"{title}\n\n{subtitle}\n\n{content}",
            link=self.driver.current_url,
            author=user.id,
            platform="medium"
        )
        
        # Save to MongoDB
        doc.save()
        
        return [doc]
```

**Key Concepts**:
- **BeautifulSoup**: HTML parsing
- **Document Creation**: Convert HTML to structured data
- **Automatic Persistence**: Save directly to MongoDB

---

### 4. `llm_engineering/application/crawlers/github.py`

**Purpose**: Clone and parse GitHub repositories.

```python
# llm_engineering/application/crawlers/github.py

import git
import os
from pathlib import Path

class GitHubCrawler(BaseCrawler):
    def extract_content(self, user, **kwargs) -> list:
        """Extract GitHub repository content"""
        repo_url = self.driver.current_url
        
        # Clone repository
        temp_dir = Path("temp_repos") / str(uuid.uuid4())
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Clone with GitPython
            repo = git.Repo.clone_from(repo_url, temp_dir)
            
            # Read all Python files
            documents = []
            for file_path in temp_dir.rglob("*.py"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = RepositoryDocument(
                    content=content,
                    link=str(file_path),
                    author=user.id,
                    platform="github",
                    metadata={
                        "repo": repo_url,
                        "file": str(file_path.relative_to(temp_dir))
                    }
                )
                documents.append(doc)
            
            # Bulk save
            RepositoryDocument.bulk_insert(documents)
            
            return documents
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
```

**Key Concepts**:
- **GitPython**: Programmatic Git operations
- **Temporary Files**: Clone to temp directory
- **Recursive File Search**: `rglob("*.py")`

---

## 🔧 Pipeline Integration

### ETL Pipeline

```python
# pipelines/digital_data_etl.py

from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user

@pipeline
def digital_data_etl(user_full_name: str, links: list[str]):
    """
    ETL pipeline to collect data from various sources.
    
    Args:
        user_full_name: Name of the user
        links: List of URLs to crawl
    """
    # Step 1: Get or create user document
    user = get_or_create_user(user_full_name)
    
    # Step 2: Crawl all links
    crawl_links(user=user, links=links)
```

### Crawl Links Step

```python
# steps/etl/crawl_links.py

from zenml import step
from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher

@step
def crawl_links(user: UserDocument, links: list[str]) -> None:
    """Crawl multiple links using appropriate crawlers"""
    
    # Build dispatcher
    dispatcher = CrawlerDispatcher.build()
    
    # Crawl each link
    for link in links:
        crawler = dispatcher.get_crawler(link)
        documents = crawler.extract(link=link, user=user)
        
        logger.info(f"Crawled {link}: extracted {len(documents)} documents")
```

---

## 🛠️ Hands-On: Create Custom Crawler

### Task: Crawl a Blog

Let's create a crawler for a generic blog.

#### Step 1: Create Crawler Class

```python
# llm_engineering/application/crawlers/my_blog.py

from bs4 import BeautifulSoup
from llm_engineering.domain.documents import ArticleDocument

class MyBlogCrawler(BaseCrawler):
    def extract_content(self, user, **kwargs) -> list:
        """Extract blog post content"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # Inspect the blog's HTML structure first
        # Adjust selectors based on actual HTML
        
        # Example: Extract title
        title = soup.find('h1', class_='entry-title')
        title = title.text.strip() if title else "Unknown"
        
        # Extract content
        content_div = soup.find('div', class_='entry-content')
        if content_div:
            paragraphs = content_div.find_all('p')
            content = '\n'.join([p.text.strip() for p in paragraphs])
        else:
            content = ""
        
        # Create document
        doc = ArticleDocument(
            content=f"{title}\n\n{content}",
            link=self.driver.current_url,
            author=user.id,
            platform="my_blog"
        )
        
        doc.save()
        return [doc]
```

#### Step 2: Register in Dispatcher

```python
# In CrawlerDispatcher
def register_my_blog(self):
    from .my_blog import MyBlogCrawler
    self._crawlers["myblog.com"] = MyBlogCrawler
```

---

## 🐛 Error Handling & Retries

### Robust Crawler Implementation

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustCrawler(BaseCrawler):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def extract(self, link: str, **kwargs) -> list:
        """Extract with automatic retries"""
        try:
            self.driver.get(link)
            
            # Wait for page load
            self.driver.implicitly_wait(10)
            
            # Check for errors
            if "404" in self.driver.title:
                logger.error(f"404 error for {link}")
                return []
            
            return self.extract_content(**kwargs)
            
        except Exception as e:
            logger.error(f"Error extracting {link}: {e}")
            raise  # Trigger retry
```

**Key Concepts**:
- **Retry Logic**: Automatic retries on failure
- **Exponential Backoff**: Wait longer between retries
- **Implicit Waits**: Wait for elements to load

---

## 📊 Testing Your Crawler

### Unit Test Example

```python
# tests/unit/test_crawlers.py

import pytest
from llm_engineering.application.crawlers.medium import MediumCrawler

@pytest.mark.skip(reason="Slow integration test")
def test_medium_crawler():
    crawler = MediumCrawler()
    documents = crawler.extract(
        link="https://medium.com/example-article",
        user=test_user
    )
    
    assert len(documents) > 0
    assert documents[0].platform == "medium"
    assert len(documents[0].content) > 0
```

---

## 🎯 Best Practices

### 1. **Respect Robots.txt**

```python
# Check robots.txt before crawling
import requests

def can_crawl(base_url: str) -> bool:
    robots_url = f"{base_url}/robots.txt"
    response = requests.get(robots_url)
    
    if "Disallow: /" in response.text:
        return False
    
    return True
```

### 2. **Rate Limiting**

```python
import time
from random import uniform

def respectful_crawl(urls: list[str]):
    for url in urls:
        crawler.extract(link=url)
        
        # Random delay between requests (2-5 seconds)
        time.sleep(uniform(2, 5))
```

### 3. **User Agent Rotation**

```python
from fake_useragent import UserAgent

ua = UserAgent()
options.add_argument(f"user-agent={ua.random}")
```

---

## 📝 Exercise: Build a Crawler

### Task

Create a crawler for your favorite website (blog, news site, etc.).

### Requirements

1. Inherit from `BaseCrawler`
2. Implement `extract_content()`
3. Extract title, content, and metadata
4. Save to MongoDB
5. Register in dispatcher
6. Test with a real URL

### Template

```python
from llm_engineering.application.crawlers.base import BaseCrawler
from llm_engineering.domain.documents import ArticleDocument

class MyCustomCrawler(BaseCrawler):
    def extract_content(self, user, **kwargs) -> list:
        # Your code here
        pass
```

---

## 🎓 Knowledge Check

1. **What pattern does CrawlerDispatcher use?**
   - Answer: Strategy Pattern

2. **Why use headless Chrome?**
   - Answer: Run on servers without display

3. **What does chromedriver_autoinstaller do?**
   - Answer: Automatically installs matching ChromeDriver

4. **How are documents persisted?**
   - Answer: Via `doc.save()` to MongoDB

---

## 🔗 Next Session

**Session 2.2**: Text Preprocessing Pipeline

We'll learn about:
- Text cleaning with regex
- Chunking strategies
- Handler pattern for processing

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 1.1, Basic HTML knowledge

**Outcome**: You'll be able to create custom web crawlers for any website.
