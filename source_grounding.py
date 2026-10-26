"""Fetch original-source context and validate the separate editorial verdict."""
from html.parser import HTMLParser
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
import json
import re

ALLOWED_HOSTS = {'openai.com','www.openai.com','huggingface.co','simonwillison.net',
    'krebsonsecurity.com','www.bleepingcomputer.com','www.microsoft.com',
    'unit42.paloaltonetworks.com','www.darkreading.com','www.mandiant.com',
    'cloud.google.com','blog.google','www.cisa.gov'}

def check_source_url(url):
    p=urlsplit(url)
    if p.scheme!='https' or p.hostname not in ALLOWED_HOSTS or p.username or p.password or p.port not in (None,443):
        raise ValueError('Source context URL is outside the approved public feed hosts')
    return url

class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_source_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

class SourceText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[]; self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','nav','footer','header','form','aside','svg'):
            self.stack.append(tag)
        if not self.stack and tag in ('p','h1','h2','h3','li','article','main'):
            self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in self.stack:
            self.stack=self.stack[:self.stack.index(tag)]
    def handle_data(self,data):
        if not self.stack: self.parts.append(data)
    def text(self):
        return '\n'.join(' '.join(line.split()) for line in ''.join(self.parts).splitlines() if line.strip())

def fetch_source_context(url):
    check_source_url(url)
    request=Request(url,headers={'User-Agent':'Mozilla/5.0 (source-backed editorial reader)'})
    with build_opener(PublicRedirects()).open(request,timeout=25) as response:
        check_source_url(response.geturl())
        raw=response.read(3_000_001)
    if len(raw)>3_000_000: raise ValueError('Source exceeds the response budget')
    parser=SourceText(); parser.feed(raw.decode('utf-8',errors='replace'))
    text=parser.text()[:18000]
    if len(text.split())<100: raise ValueError('Insufficient original source content for grounded writing')
    return text

def validate_verdict(raw):
    result=json.loads(raw)
    if not isinstance(result,dict) or result.get('publishable') is not True:
        raise ValueError('Editorial review rejected the article: '+str(result.get('reasons',[]) if isinstance(result,dict) else 'invalid verdict'))
    for field in ('grounded','specific','non_generic'):
        if result.get(field) is not True: raise ValueError('Editorial review failed '+field)
    return result

def assert_supported_identifiers(body,source):
    invented=set(re.findall(r'\bCVE-\d{4}-\d{4,}\b',body,re.I))-set(re.findall(r'\bCVE-\d{4}-\d{4,}\b',source,re.I))
    if invented: raise ValueError('Draft contains CVE identifiers not present in original source')
