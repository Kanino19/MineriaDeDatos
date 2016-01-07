f = open('german.data-numeric','r')
line = f.read()		
caden = line.split('\n')
n =len(caden)		
for i in range(n):
	caden[i] = caden[i].split(' ')
	m =len(caden[i])
	k = 0
	while k < m:
		if caden[i][k] == '':
			del(caden[i][k])
			m = len(caden[i])
		else:		
			k += 1

for j in range(n):
	caden[j] = ' '.join(caden[j])
line ='\n'.join(caden)
f.close()
g = open('german.data-numeric','w')
g.write(line)
g.close()
