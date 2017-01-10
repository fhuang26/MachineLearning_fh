import numpy as np
import pywt
import matplotlib.pyplot as plt

wavelettype = 'haar'

def generate_feature_vectors(TS):
	TSwfv = {} # Wavelet feature vector
	TSsse = {} # Sum of Square Errors
	TSrec = {} # reconstructed time series
	for TSname in TS:
		cL2 = pywt.wavedec(TS[TSname],wavelettype,level=2)
		cL4 = pywt.wavedec(TS[TSname],wavelettype,level=4)
		cL6 = pywt.wavedec(TS[TSname],wavelettype,level=6)
		cL8 = pywt.wavedec(TS[TSname],wavelettype,level=8)
		cL10 = pywt.wavedec(TS[TSname],wavelettype,level=10)
		# Generate array of feature vectors for this time series
		TSwfv[TSname] = [list(cL2[0])+list(cL2[1]), \
			list(cL4[0])+list(cL4[1]), \
			list(cL6[0])+list(cL6[1]), \
			list(cL8[0])+list(cL8[1]), \
			list(cL10[0])+list(cL10[1])]
		#print len(TSwfv[TSname][0])
		#print len(TSwfv[TSname][1])
		#print len(TSwfv[TSname][2])

		#print "TSwfv[{0}]:{1}".format(TSname,TSwfv[TSname])

		# Generate array of SSE values for this time series
		TSsse[TSname] = []

		# Level 2
		# print len(cL2), len(cL4), len(cL6), len(cL8), len(cL10)
		c = [cL2[0],cL2[1]]
		for i in range(2,len(cL2)):
			c.append([0]*len(cL2[i]))
		TSrecL2 = pywt.waverec(c,wavelettype)
		sse = sum([(TS[TSname][i]-TSrecL2[i])**2 for i,v in enumerate(TSrecL2)])
		TSsse[TSname].append(sse)

		# Level 4
		c = [cL4[0],cL4[1]]
		for i in range(2,len(cL4)):
			c.append([0]*len(cL4[i]))
		TSrecL4 = pywt.waverec(c,wavelettype)
		sse = sum([(TS[TSname][i]-TSrecL4[i])**2 for i,v in enumerate(TSrecL4)])
		TSsse[TSname].append(sse)

		# Level 6
		c = [cL6[0],cL6[1]]
		for i in range(2,len(cL6)):
			c.append([0]*len(cL6[i]))
		TSrecL6 = pywt.waverec(c,wavelettype)
		sse = sum([(TS[TSname][i]-TSrecL6[i])**2 for i,v in enumerate(TSrecL6)])
		TSsse[TSname].append(sse)

		# Level 8
		c = [cL8[0],cL8[1]]
		for i in range(2,len(cL8)):
			c.append([0]*len(cL8[i]))
		TSrecL8 = pywt.waverec(c,wavelettype)
		sse = sum([(TS[TSname][i]-TSrecL8[i])**2 for i,v in enumerate(TSrecL8)])
		TSsse[TSname].append(sse)

		TSrec[TSname] = [TSrecL2,TSrecL4,TSrecL6,TSrecL8]

		# Plotting
		if TSname == 'TS.AMZN':
			print [len(x) for x in cL2]
			print [len(x) for x in cL4]
			print [len(x) for x in cL6]
			print [len(x) for x in cL8]
			lenL2 = str(len(cL2[0])+len(cL2[1]))
			lenL4 = str(len(cL4[0])+len(cL4[1]))
			lenL6 = str(len(cL6[0])+len(cL6[1]))
			lenL8 = str(len(cL8[0])+len(cL8[1]))
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_xlabel('Time index (1 = 4/20/2011)')
			ax.set_ylabel('Closing price (normalized)')
			ax.plot(TS[TSname],label=TSname)
			LL2 = TSname+' (L2) len: '+lenL2+' SSE: '+'%3.2f'%TSsse[TSname][0]
			LL4 = TSname+' (L4) len: '+lenL4+' SSE: '+'%3.2f'%TSsse[TSname][1]
			LL6 = TSname+' (L6) len: '+lenL6+' SSE: '+'%3.2f'%TSsse[TSname][2]
			LL8 = TSname+' (L8) len: '+lenL8+' SSE: '+'%3.2f'%TSsse[TSname][3]
			ax.plot(TSrecL2,label=LL2)
			ax.plot(TSrecL4,label=LL4)
			ax.plot(TSrecL6,label=LL6)
			ax.plot(TSrecL8,label=LL8)
			plt.legend(prop={'size':10},loc=0)
			plt.savefig(TSname+'.pdf',edgecolor='b', format='pdf')

	return TSwfv,TSsse,TSrec