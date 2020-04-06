from bs4 import BeautifulSoup
import codecs
import urllib.request

#try:
#    data_file=open("C:\Users\Arno\PycharmProjects\KI\var6_D_46_regression.html","r")
#    data_list=data_file.readlines()
#finally:
#    data_file.close()

f=codecs.open("var6_D_46_regression.html",'r', 'utf-16')
document= BeautifulSoup(f.read(),'html.parser')#.get_text()
print(document)


#page = urllib.request.urlopen("C:\Users\Arno\PycharmProjects\KI\var6_D_46_regression.html").read()
#print(page)