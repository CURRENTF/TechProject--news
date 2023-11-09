import urllib.request
from bs4 import BeautifulSoup


def get_web_content(link: str):
    Page = urllib.request.urlopen(link)
    PageHtml = Page.read().decode('utf-8')
    return PageHtml


def web_content_to_json(html):
    text_json = {}

    pageSoup = BeautifulSoup(html, 'html.parser')
    pageInfo = pageSoup.find_all('div', class_='article')[0]
    h3 = pageInfo.find('h3')
    h1 = pageInfo.find('h1')
    h2 = pageInfo.find('h2')

    # 标题
    title = h3.text.strip() + h1.text.strip() + h2.text.strip()

    # 处理paragraphs
    article = pageInfo.find('div', id='ozoom')
    paragraphs = article.find_all('p')
    paragraphs = [p.text.strip() for p in paragraphs]

    text_json['title'] = title
    text_json['paragraphs'] = paragraphs

    return text_json
