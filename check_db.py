f = open("../human.fasta")
cnt = -1
found = False
buff = ""
for l in f:
    if l[0] == ">":
        cnt += 1
        if found:
            break
    if 'Q04727' in l:
        print("FOUND AT {}".format(cnt))
        found=True
    else:
        if "MYPQG" in l:
            print("Found the substring at: {}".format(cnt))
        if found:
            buff += l.strip()

print(buff)
