# import scrapy

# https://newscatcherapi.com/blog/python-web-scraping-libraries-to-mine-news-data
# class QuotesSpider(scrapy.Spider):
#     name = 'quotes'
#     start_urls = [
#         'https://quotes.toscrape.com/tag/humor/',
#     ]

#     def parse(self, response):
#         for quote in response.css('div.quote'):
#             yield {
#                 'author': quote.xpath('span/small/text()').get(),
#                 'text': quote.css('span.text::text').get(),
#             }

#         next_page = response.css('li.next a::attr("href")').get()
#         if next_page is not None:
#             yield response.follow(next_page, self.parse)

# from newspaper import New, describe_url
# import json
# import time

# nyt = Newscatcher(website = 'nytimes.com')
# results = nyt.get_news()

# count = 0
# articles = results['articles']
# for article in articles[:10]:   
#    count+=1
#    print(
#      str(count) + ". " + article["title"] \
#      + "\n\t\t" + article["published"] \
#      + "\n\t\t" + article["link"]\
#      + "\n\n"
#      )
#    time.sleep(0.33)


 # %%
from typing import Union
import newspaper
from newspaper import Article
import json
 # %%


 # %%

# http://www.cnn.com/2013/11/27/justice/tucson-arizona-captive-girls/
# http://www.cnn.com/2013/12/11/us/texas-teen-dwi-wreck/index.html
# ...

# >>> for category in cnn_paper.category_urls():
# >>>     print(category)

# http://lifestyle.cnn.com
# http://cnn.com/world
# http://tech.cnn.com
# ...

# >>> cnn_article = cnn_paper.articles[0]
# >>> cnn_article.download()
# >>> cnn_article.parse()
# >>> cnn_article.nlp()
# ...
# %%

# %%
def clean_url(url:str):
    url = url[11:]
    url =  url.replace("-","_")
    return re.sub("[^a-zA-Z0-9_ ]", "", url)

# %%
def make_file_name(url):
    url2 = clean_url(url)
    max_length = 50
    if len(url2)> max_length:
        url2 = url2[0:max_length]
    return r"news/" +url2 + ".json"
# %%
def make_article_save_name(article):
    return make_file_name(article.url)

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
    # %%

from pathlib import Path
import re
news_folder = Path("news")
news_folder.mkdir(exist_ok=True)
# %%



scrape_news = False

if scrape_news:
    news_website = newspaper.build('https://www.nbcnews.com/')
    # for article in news_website.articles:
    #     print(article.url)
    for article in news_website.articles:
        article.download()
        article.parse()
    for article in news_website.articles:
        with open(make_article_save_name(article), 'w') as f:
            json.dump(vars(article),f,default=str)

if scrape_news == False:
    news_website = newspaper.Source('https://www.nbcnews.com/')


    json_save_news = news_folder.glob("*.json")
    for json_file in json_save_news:
        with open(json_file,'r') as f:
            article = DictObj(json.load(f))
        news_website.articles.append(article)
    print("num saved objects is ",len(news_website.articles))
    
        # article.download()
        # article.parse()

# %%
from transformers import pipeline
# from getsampletxt import get_sample_txt

# %%

qa_model = pipeline("question-answering", model = "deepset/roberta-base-squad2")
question = "is someone sick"
# %%

# def get_text(something: Union[Article, dict], ):
#     if type(something) == dict:
#         return something['text']
#     if type(something) == Article:
#         return something.text
#     raise Exception("Unkown type")

for i,a in enumerate(news_website.articles):
    text = a.text
    if len(text)<100:
        continue
    context = text
    res = qa_model(question = question, context = context)
    a.model_response = res
    if res['score']<0.1:
        # no good answer
        print(f"No good answer for {i}, {a.title}")
        continue
    print(f"answer = {res['answer']}, {i}, {res}, {a.title}")

# %%
