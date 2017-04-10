#!/usr/bin/env python

import sys
import time
import urllib2
quote = sys.argv[1]
date = sys.argv[2]

print 'Scraping data for {0} on {1}'.format(quote, date)

date_time = date + " 9:30:00"

epoch = int(time.mktime(time.strptime(date_time, "%Y%m%d %H:%M:%S")))

url = 'https://www.google.com/finance/getprices?q={0}&x=NASD&i=60&p=1d&f=d,c,v,o,h,l&df=cpct&auto=1&ts={1}'.format(quote.upper(), epoch)

r = urllib2.urlopen(url).read()

filename = '{0}_{1}.csv'.format(quote.upper(), date)
with open(filename, 'w') as f:
	f.write(r)

