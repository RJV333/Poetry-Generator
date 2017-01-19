# -*- coding: utf-8 -*-
import sys, re
import nltk
from nltk.corpus import treebank
from nltk.corpus import cmudict
from nltk.tag.util import untag  # Untags a tagged sentence.
from collections import defaultdict
from math import log, exp
import copy
import random



toremove = [',',':',';','.','!',"'","--","?", '']
unknown_token = "<UNK>"  # unknown word token.

linelength = 8
branch_factor = 2
poem_length = 8 #make divisible by 2 for couplets, 4 for alt rhymes
pndct = cmudict.dict()

def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]
                
class poem_model:
	def __init__(self, verse_lines, tagged_lines, cmudict):
		self.vl= verse_lines
		self.tvl= tagged_lines
		self.pronounciations = cmudict
		self.lastwords = self.getlastwords()
		self.tag_bigram_counts = defaultdict(float)
		self.tag_unigram_counts = defaultdict(float)
		self.reverse_transitions = defaultdict(lambda: log( float(0.0000001) ))
		self.vocab = self.getvocab(self.vl)
		self.word_tags = {}
		self.all_tags = defaultdict( lambda: set() )
		self.word_neighbors= defaultdict(lambda: set())
		self.rhymes = [] 
	def select_rhymes(self):
                result = []
                totake = poem_length
                while (totake > 0):
                        i = random.randint(0, len(self.rhymes))
                        #i = 5
                        result.append( self.rhymes[i] )
                        totake = totake -2
                return result
        def write_poemcouplets(self):
                rhymes = self.select_rhymes()
                print rhymes
                for i in range(len(rhymes) ):
                        l1 = self.makeline( rhymes[i][0] )
                        l2 = self.makeline(rhymes[i][1] )
                        l1.prntwords()
                        l2.prntwords()
        def write_poemalt(self):
                rhymes = self.select_rhymes()
                print rhymes
                for i in range(0,len(rhymes)-1,2):
                        l1 = self.makeline( rhymes[i][0] )
                        l1.prntwords()
                        l2 = self.makeline( rhymes[i+1][0] )
                        l2.prntwords()
                        l3 = self.makeline( rhymes[i][1] )
                        l3.prntwords()
                        l4 = self.makeline( rhymes[i+1][1] )
                        l4.prntwords()                                                
                
        def enquelines(self, verse_line):
                verse_line.updatephenomes()
                posslists  = []
                #print "starters"
                for prevword in self.word_neighbors[verse_line.words[0][0] ]:
                       #print prevword
                        if prevword in self.pronounciations and prevword not in verse_line.wordset:
                                #print prevword
                                sound_prob = self.sound_likelihood(prevword, verse_line.phenome_set)
                                #print sound_prob
                                for tag in self.all_tags[prevword ]:
                                        #print tag
                                        tag_prob = self.reverse_transitions[ tag, verse_line.words[0][1] ]
                                        #print tag_prob
                                        new_line = copy.deepcopy( verse_line )
                                        self.updatenewline( prevword, tag, new_line, sound_prob,tag_prob)
                                        if new_line.syllable_length <= linelength:
                                                posslists.append(new_line)
                posslists.sort(key=lambda x: x.prob, reverse=True)
                if len(posslists) > branch_factor:
                        posslists = posslists[:branch_factor]
                return posslists
                
        def makeline(self, end_word):
                queue = []
                keepers = []
                for tag in self.all_tags[end_word]:
                        end_line = line( (end_word, tag))
                        end_line.updatephenomes()
                        queue.append(end_line)
                while len(queue) != 0 :
                        if queue[0].syllable_length < linelength:
                                more_lines = self.enquelines ( queue[0] )
                        for l in more_lines:
                                if l.syllable_length < linelength:
                                        queue.append(l)
                                if l.syllable_length ==linelength:
                                        keepers.append(l)
                        k = queue.pop(0)
                keepers.sort(key=lambda x: x.prob, reverse=True)
                return keepers[0]

        def updatenewline(self, prevword, tag, new_line, soundprob, tagprob ):
                new_line.words.insert ( 0, (prevword, tag) )
                new_line.updatephenomes()
                new_line.line_syll_length()
                new_line.prob = new_line.prob + soundprob + tagprob
                new_line.wordset.add(prevword)
                

        def get_rev_transitions(self, training_set):
                self.trans_grams(training_set)
                self.compute_rev_trans()
        def trans_grams(self, training_set):
                for sentence in training_set:
                        for (word, tag) in sentence:
                                self.tag_unigram_counts[tag] += 1
                for sentence in training_set:
                        for i in range (0, len(sentence) - 1):
                                self.tag_bigram_counts[ (sentence[i][1], sentence[i+1][1] ) ] +=1                
        def compute_rev_trans(self):
                for (tag1, tag2) in self.tag_bigram_counts:
                        self.reverse_transitions[ (tag1, tag2) ] = log( float( self.tag_bigram_counts[ (tag1, tag2) ] ) / float( self.tag_unigram_counts[ (tag2) ] ) )
        def sound_likelihood(self, word, soundset):
                phoneme_list = self.pronounciations[word][0]
                phcount = 0
                #print phoneme_list
                for i in range(len(phoneme_list)):
                        if phoneme_list[i] in soundset:
                                phcount+=1
                if phcount == 0:
                        phcount = .01
                result = float(phcount) / float( len(soundset) )
                result = log(result)
                return result
                
        def most_likely_tags(self):
                for line in self.tvl:
                        for word in line:
                                if word[0] not in self.word_tags:
                                        self.word_tags[word[0]] = {word[1]:1}
                                elif word[1] not in self.word_tags[word[0]]:
                                        self.word_tags[word[0]][word[1]] = 1
                                else:
                                        self.word_tags[word[0]][word[1]]+=1
        def get_all_tags(self):
                for line in self.tvl:
                        for word in line:
                                self.all_tags[word[0] ].add(word[1])
                                        
        def getvocab(self, sentence_set):

                vocab = set()
                for line in sentence_set:
                        for k in line:
                                if k not in vocab:
                                        vocab.add(k)
                return vocab
        def makeneighbors(self):
                for line in self.vl:
                        for word in line:
                                for w in line:
                                        if w != word:
                                                self.word_neighbors[word].add(w)
        def getlastwords(self):
                result = []
                for line in self.vl:
                        if len(line) >0:
                                result.append( line[-1] )
                return result          
        def getrhymes(self):
                result = []
                for i in range(len(self.lastwords)-4):
                        for j in range(1,4):
                                if possrhyme(self.lastwords[i], self.lastwords[i+j], self.pronounciations )[0]:
                                        emp = possrhyme(self.lastwords[i], self.lastwords[i+j], self.pronounciations )[1]
                                        #print "HERE", self.lastwords[i], self.lastwords[i+j]
                                        if emp !="":
                                                if confirm_rhyme(self.lastwords[i], self.lastwords[i+j], emp, self.pronounciations ):
                                                        #print "CONFIRMED", self.lastwords[i], self.lastwords[i+j]
                                                        result.append( ( self.lastwords[i], self.lastwords[i+j]) )
                self.rhymes = result
                
    
class line:
        def __init__(self, endword):
                #self.pronounce = pd
                #self.last_word=endword
                self.syllable_length = num_syll_word(endword)
                self.words = [endword]
                self.wordset = set()
                self.wordset.add(endword)
                self.phenome_set = self.updatephenomes()
                self.prob = 1
        def prntwords(self):
                for i in range(len(self.words)):
                        print self.words[i][0],
                print "\n"
        def prntwordtags(self):
                for i in range(len(self.words)):
                        print self.words[i]
                print "\n"
        def line_syll_length(self):
                syl = 0
                for word in self.words:
                        syl = syl + num_syll_word(word)
                self.syllable_length = syl                     
        
        def updatephenomes(self):
                result = set()
                for word in self.words:
                        for pronounce in pndct[word[0]]:
                                for phen in pronounce:
                                        result.add(phen)
                self.phenome_set = result

def num_syll_word(word):
        syl =0
        for phen in pndct[word[0]][0]:
                #print phen, phen[-1]
                if phen[-1] == '0' or phen[-1] == '1' or phen[-1] == '2':
                        syl+=1
        return syl

def confirm_rhyme(word1,word2, emp, prondict):

        w1phenomes = prondict[word1][0]
        w2phenomes = prondict[word2][0]
        x = len(w1phenomes)-1
        y = len(w2phenomes)-1
        w1in= w1phenomes[x]
        w2in= w2phenomes[y]
        
        while(w1in!=emp):
                x= x-1
                w1in= w1phenomes[x]
        
        while(w2in!=emp):
                y = y-1
                w2in = w2phenomes[y]
        
        while( x!=len(w1phenomes) and y!=len(w2phenomes) ):
                if w1phenomes[x] != w2phenomes[y]:
                        return False
                x+=1
                y+=1
        if x != len(w1phenomes) or y != len(w2phenomes):
                return False
        return True
        
        
def possrhyme(word1, word2, prondict):
        emp1 = lastemphasis(word1, prondict)
        emp2 = lastemphasis(word2, prondict)
        if emp1 == emp2:
                return (True, emp1)
        else: 
                return (False, emp1)
def lastemphasis(word, prondict):
        lastemph = ""
        if word in prondict:
                pronounciations = prondict[word]
                if len(pronounciations) == 1:
                        for pronounce in pronounciations:
                                for i in range( len(pronounce) ):
                                        if pronounce[i][-1] == '1' or pronounce[i][-1] == '2':
                                                lastemph = pronounce[i]
        return lastemph
        
def bypoems(vl):
        result = []
        poemdone = False
        poemtoadd = []
        for line in vl:
                if len(line) == 0:
                        result.append(poemtoadd)
                        poemtoadd = []
                else:
                        poemtoadd.append(line)        
        for poem in result:
                if len(poem) == 0:
                        result.remove(poem)
        return result

def parse(vl):
        for line in vl:
                for word in line:
                        if word in toremove:
                                line.remove(word)
def lowercase(vl):
        for line in vl:
                for i in range(len(line)):
                        line[i] = line[i].lower()
def hiddenchars(vl):
        for line in vl:
                if len(line)>0:
                        for tok in toremove:
                                word = line.pop()
                                line.append( word.replace(tok,"") )
def taglines(vl):
        taggedvl = []
        for line in vl:
            if len(line) > 0 and line[-1] == '':
               line.pop(len(line)-1)
            taggedvl.append( nltk.pos_tag(line) )
        return taggedvl


def main():
	t = open('19221.txt', 'rU')
	verse_lines = t.read()
	verse_lines = verse_lines.split('\n')
	vl = []
	for verse in verse_lines:
		vl.append( nltk.word_tokenize(verse) )
		
	bypoems(vl)
	parse(vl)
	lowercase(vl)
	hiddenchars(vl)
	taggedvl = taglines(vl)
	d = cmudict.dict()
	
	pm = poem_model(vl, taggedvl, d)               
        treebank_tagged_sents = TreebankNoTraces()
        training_set = treebank_tagged_sents[:3000]
        pm.get_rev_transitions(training_set)
        pm.most_likely_tags()
        pm.get_all_tags()
        pm.makeneighbors()
        pm.getrhymes()
        
        print "thanks"
	return pm