import json
import re
import glob
import random
import os

# Change to the task name
task_name = "kg_task_real_multiple_linked"

path_normalized = './new_data/'+task_name+'_normalized/'
path_original = './new_data/'+task_name+'_original/'

try: # beacause the path might already exist
	os.makedirs(path_normalized)
	os.makedirs(path_original)
except:
	pass

train_file = open('./new_data/'+task_name+'_normalized/'+task_name+'_normalized_train.txt', 'w')
train_original_file = open('./new_data/'+task_name+'_original/'+task_name+'_original_train.txt', 'w')
test_file = open('./data/'+task_name+'_normalized/'+task_name+'_normalized_test.txt', 'w')
test_original_file = open('./new_data/'+task_name+'_original/'+task_name+'_original_test.txt', 'w')
train_file_list = open('./new_data/'+task_name+'_original/'+task_name+'_train_file_list.txt', 'w')
test_file_list = open('./new_data/'+task_name+'_original/'+task_name+'_original_test_file_list.txt', 'w')

# The directory to input files in the json format
json_files = glob.glob("./data/sample_json_files/*.json")

n_kgs = len(json_files)
n_kgs_train = int(n_kgs * 0.9)
# for test only:
# n_kgs_train = -1
counter_number = 0

for j, json_file in enumerate(json_files):

    json_data = open(json_file)
    data = json.load(json_data, encoding="utf-8")
    # Using the entity2id instead of having different lists because we are supposed that we have positional encodings!
    entity2id = {}
    my_list = list(range(1, 3000))  # list of integers from 1 to 1000 (1000 is max length of kg*3)
    entity2id_list = random.sample(my_list, len(my_list))

    if j <= n_kgs_train:
        destination_normalized_file = train_file
        destination_original_file = train_original_file
        train_file_list.write(json_file+'\n')

    else:
        destination_normalized_file = test_file
        destination_original_file = test_original_file
        test_file_list.write(json_file + '\n')

    index = 0
    for i, item in enumerate(data['OriginalAxioms']):

        line = str(i + 1) + ' '

        #raw_line =  re.split("( |\\\"|'.*?\\\"|'.*?')", item.encode('utf-8'))
        raw_line = re.split(''' (?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', (item.encode('utf-8')).replace('"', ''))


        # To deal with few instance that our regular expression is spliting bad for us
        if len(raw_line) > 3:

            raw_line = raw_line[0:2] + [" ".join(raw_line[2:])]


	original_line = line + ' '.join(raw_line) # To add the line number to the resulted output
	print (original_line)

        for k, elem in enumerate(raw_line):

            if elem.startswith("rdf:") or elem.lstrip().startswith("rdfs:"):
                line = line + str(elem) + ' '

            else:

                if entity2id.has_key(elem):
                    line = line + entity2id[elem] + ' '
                else:
                    entity2id[elem] = 'e'+ str(entity2id_list[index])
                    index += 1
                    line = line + entity2id[elem] + ' '


        print (line+'\n')
        destination_normalized_file.write(
            line.encode('utf-8') + '\n')
        destination_original_file.write(
            (original_line) + '\n')

    for i, item in enumerate(data['InferredAxioms']):
        line = str(i + 1+len(data['OriginalAxioms'])) + ' '
        # raw_line =  re.split("( |\\\"|'.*?\\\"|'.*?')", item.encode('utf-8'))
        raw_line = re.split(''' (?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', (item.encode('utf-8')).replace('"', ''))

        if len(raw_line) > 3:

            raw_line = raw_line[0:2] + [" ".join(raw_line[2:])]

        
	original_line = line + ' '.join(raw_line) # To add the line number to the resulted output
	print (original_line)

        for k, elem in enumerate(raw_line):

            if elem.startswith("rdf:") or elem.lstrip().startswith("rdfs:"):
                line = line + str(elem) + ' '

            else:

                if entity2id.has_key(elem):
                    #print  "infered-1"
                    line = line + entity2id[elem] + ' '
                else:
                    print (elem)
                    print ("infered-0")
                    print ("json_file_name= ",json_file)

                    try:
                        entity2id[elem] = 'e' + str(entity2id_list[index])
                        index += 1
                        line = line + entity2id[elem] + ' '
                    except:
                        print ("index=", index)
                        print ("name_of_file=", json_file)
                        counter_number += 1
                        pass

        print(line + '\tyes\n')

        destination_normalized_file.write(
            line.encode('utf-8') + '\tyes\n')
        destination_original_file.write(
            original_line + '\tyes\n')

    for i, item in enumerate(data['InvalidAxioms']):
        line = str(i + 1+len(data['OriginalAxioms'])+len(data['InferredAxioms'])) + ' '
        # raw_line =  re.split("( |\\\"|'.*?\\\"|'.*?')", item.encode('utf-8'))
        raw_line = re.split(''' (?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', (item.encode('utf-8')).replace('"', ''))

        if len(raw_line) > 3:
            raw_line = raw_line[0:2] + [" ".join(raw_line[2:])]


	original_line = line + ' '.join(raw_line) # To add the line number to the resulted output
	print (original_line)

        for k, elem in enumerate(raw_line):

            if elem.startswith("rdf:") or elem.lstrip().startswith("rdfs:"):
                line = line + str(elem) + ' '

            else:

                if entity2id.has_key(elem):
                    line = line + entity2id[elem] + ' '
                else:
                    print (elem)
                    print ('InvalidAxioms-0')
                    print ("json_file_name= ", json_file)

                    try:
                        entity2id[elem] = 'e' + str(entity2id_list[index])
                        line = line + entity2id[elem] + ' '
                        index += 1
                    except:
                        print ("index=", index)
                        print ("name_of_file=",json_file)

                        counter_number += 1
                        pass

        print (line + '\tno\n')

        destination_normalized_file.write(
            line.encode('utf-8') + '\tno\n')
        destination_original_file.write(
            original_line + '\tno\n')

print ("counter_number = ", counter_number)
train_file.close()
test_file.close()

