#!/usr/bin/env python

import sys
import time, calendar
import urllib2
import json

quote = sys.argv[1]
date = sys.argv[2]

print 'Scraping data for {0} on {1}'.format(quote, date)

#epoch = calendar.timegm(time.gmtime(time.mktime(time.strptime(date, "%Y%m%d"))))
#print epoch

url = 'https://query2.finance.yahoo.com/v7/finance/options/AAPL?date=1494547200'
#url = "https://query2.finance.yahoo.com/v7/finance/options/{0}?date={1}".format(quote, epoch)
#print url

r = urllib2.urlopen(url)

data = json.load(r)

calls = data['optionChain']['result'][0]['options'][0]['calls']
puts = data['optionChain']['result'][0]['options'][0]['puts']

hdr = []
for key in calls[0].iterkeys():
    hdr.append(key)


filename = '{0}_{1}_call_option.csv'.format(quote.upper(), date)
with open(filename, 'w') as f:
    f.write(','.join(hdr))
    f.write('\n')

    line = []
    for call in calls:
        for field in hdr:
            line.append(str(call[field]))

        f.write(','.join(line))
        f.write('\n')
        line = []


hdr = []
for key in puts[0].iterkeys():
    hdr.append(key)


filename = '{0}_{1}_put_option.csv'.format(quote.upper(), date)
with open(filename, 'w') as f:
    f.write(','.join(hdr))
    f.write('\n')

    line = []
    for put in puts:
        for field in hdr:
            line.append(str(put[field]))

        f.write(','.join(line))
        f.write('\n')
        line = []







