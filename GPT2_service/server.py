# Interface framework for building asyncio services
from starlette.applications import Starlette
from starlette.responses import UJSONResponse
# Asynchronous server implementation package
import uvicorn
# Onnx model
from onnxclass import GPT2ONNXModel
# Natural language processing package
import spacy
from pathlib import Path
import os
import gc
# Google translate python package
from google.cloud import translate_v2 as translate

# Loading ML models
en_dir = Path('/modelneren/')
es_dir = Path('/modelneres/')
de_dir = Path('/modelnerde/')
fr_dir = Path('/modelnerfr/')
it_dir = Path('/modelnerit/')
spacy.prefer_gpu()
nlp_en = spacy.load(en_dir)
nlp_de = spacy.load(de_dir)
nlp_es = spacy.load(es_dir)
nlp_fr = spacy.load(fr_dir)
nlp_it = spacy.load(it_dir)
app = Starlette(debug=False)
response_header = {'Access-Control-Allow-Origin': '*'}
generate_count = 0
baseveterinary = 'veterinary'
basephilosophy = 'philosophy'
basemythology = 'mythology'
basefinance = 'finance'
basemusic = 'music'
baseliterature = 'literature'
basecooking = 'cooking'
basehistory = 'history'
basemedicalscience = 'medicalscience'
basescience = 'science'
basetechnology = 'technology'
baseaskdark = 'askdark'
baselanguage = 'language'
basegaming = 'gaming'
basemovies = 'movies'
basesports = 'sports'
basetravel = 'travel'
basescifi = 'scifi'
baseask = 'ask'
tolang = 'en'
pattern = re.compile(r'<([^>]*)>')
onnx_model = GPT2ONNXModel('gpt2_fp16.onnx', 'gpt2-xl', device='cuda', verbose=False)
onnx_modelbase = GPT2ONNXModel('gpt2base_fp16.onnx', 'gpt2-xl', device='cuda', verbose=False)

# Translate method
def translate_text(target, source, text):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target, source_language=source, mime_type='text/html')
    return result['translatedText']

# Get method
@app.route('/', methods=['GET'])
async def homepage(request):
# Request parameters
    params = request.query_params
    question = params.get('question', '')
    fromlang = params.get('language', '')
    text = ''
# Language selection
    if fromlang == 'en':
        docquestion = nlp_en(question)
        for ent in docquestion.ents:
            if ent.label_=="PERSON" or ent.label_=="ORG" or ent.label_=="PRODUCT" or ent.label_=="EVENT" or ent.label_=="WORK_OF_ART" or ent.label_=="LAW" or ent.label_=="NORP" or e.label_=="FAC":
                question = question.replace(ent.text , "<"+ent.text+">")
    if fromlang == 'de':
        docquestion = nlp_de(question)
        for ent in docquestion.ents:
            if ent.label_=="PER" or ent.label_=="ORG" or ent.label_=="MISC":
                question = question.replace(ent.text , "<"+ent.text+">")
    if fromlang == 'fr':
        docquestion = nlp_fr(question)
        for ent in docquestion.ents:
            if ent.label_=="PER" or ent.label_=="ORG" or ent.label_=="MISC":
                question = question.replace(ent.text , "<"+ent.text+">")
    if fromlang == 'it':
        docquestion = nlp_it(question)
        for ent in docquestion.ents:
            if ent.label_=="PER" or ent.label_=="ORG" or ent.label_=="MISC":
                question = question.replace(ent.text , "<"+ent.text+">")
    if fromlang == 'es':
        docquestion = nlp_es(question)
        for ent in docquestion.ents:
            if ent.label_=="PER" or ent.label_=="ORG" or ent.label_=="MISC":
                question = question.replace(ent.text , "<"+ent.text+">")
    question = re.sub(r'/>+/', '>', question)
    question = re.sub(r'/<+/', '<', question)
    translation = question
    translationclean = question.replace('<','').replace('>','')
    if fromlang != 'en':
        questiontotranslate = '<div>' + question.replace('<','<span class="notranslate">').replace('>','</span>') + '</div>'
        translation = translate_text(tolang, fromlang, questiontotranslate).replace('<span class="notranslate">' , '<').replace('</span>' , '>').replace('<div>' , '').replace('</div>' , '')
        questiontagged = translation
        translationspacy = translation.replace('<','').replace('>','')
        doctranslation = nlp_en(translationspacy)
        labels = list()
        indexbonus = 1
        for et in re.finditer(pattern, questiontagged):
            labels.append([et.start(0) - indexbonus + 1,et.end(0) - indexbonus - 1])
            indexbonus += 2
        for e in doctranslation.ents:
            if e.label_=="PERSON" or e.label_=="ORG" or e.label_=="PRODUCT" or e.label_=="EVENT" or e.label_=="WORK_OF_ART" or e.label_=="LAW" or e.label_=="NORP" or e.label_=="FAC":
                notadd = False
                for en in labels:
                    if en[1] == start_char or en[0] == end_char:
                        notadd = True
                    if en[0] == start_char or en[1] == end_char:
                        notadd = True
                    if start_char > en[0] and start_char < en[1]:
                        notadd = True
                    if end_char < en[1] and end_char > en[0]:
                        notadd = True
                    if start_char < en[0] and end_char > en[1]:
                        notadd = True
                if not notadd:
                    questiontagged = questiontagged.replace(e.text, '<' + e.text + '>')
        questiontagged = re.sub(r'/>+/', '>', questiontagged)
        questiontagged = re.sub(r'/<+/', '<', questiontagged)
        translation = questiontagged
        translationclean = questiontagged.replace('<','').replace('>','')
# Topic generation
    textbase = onnx_modelbase.generate(sess, run_name=runbase, length=10, prefix='<|startoftext|>' + translationclean + '/', batch_size=1, temperature=float(0.1), top_p=0.9, top_k=40, truncate='<|endoftext|>', include_prefix=False, return_as_list=True)
    if baseveterinary in textbase:
        prefixrequest = '<|startoftext|>' + baseveterinary + '/accepted/' + translation + '/'
        textbase = baseveterinary
    if basephilosophy in textbase:
        prefixrequest = '<|startoftext|>' + basephilosophy + '/accepted/' + translation + '/'
        textbase = basephilosophy
    if basemythology in textbase:
        prefixrequest = '<|startoftext|>' + basemythology + '/accepted/' + translation + '/'
        textbase = basemythology
    if basefinance in textbase:
        prefixrequest = '<|startoftext|>' + basefinance + '/accepted/' + translation + '/'
        textbase = basefinance
    if basemusic in textbase:
        prefixrequest = '<|startoftext|>' + basemusic + '/accepted/' + translation + '/'
        textbase = basemusic
    if baseliterature in textbase:
        prefixrequest = '<|startoftext|>' + baseliterature + '/accepted/' + translation + '/'
        textbase = baseliterature
    if basecooking in textbase:
        prefixrequest = '<|startoftext|>' + basecooking + '/accepted/' + translation + '/'
        textbase = basecooking
    if basehistory in textbase:
        prefixrequest = '<|startoftext|>' + basehistory + '/accepted/' + translation + '/'
        textbase = basehistory
    if basemedicalscience in textbase:
        prefixrequest = '<|startoftext|>' + basemedicalscience + '/accepted/' + translation + '/'
        textbase = basemedicalscience
    if basescience in textbase:
        prefixrequest = '<|startoftext|>' + basescience + '/accepted/' + translation + '/'
        textbase = basescience
    if basetechnology in textbase:
        prefixrequest = '<|startoftext|>' + basetechnology + '/accepted/' + translation + '/'
        textbase = basetechnology
    if baseaskdark in textbase:
        prefixrequest = '<|startoftext|>' + baseaskdark + '/accepted/' + translation + '/'
        textbase = baseaskdark
    if baselanguage in textbase:
        prefixrequest = '<|startoftext|>' + baselanguage + '/accepted/' + translation + '/'
        textbase = baselanguage
    if basegaming in textbase:
        prefixrequest = '<|startoftext|>' + basegaming + '/accepted/' + translation + '/'
        textbase = basegaming
    if basemovies in textbase:
        prefixrequest = '<|startoftext|>' + basemovies + '/accepted/' + translation + '/'
        textbase = basemovies
    if basesports in textbase:
        prefixrequest = '<|startoftext|>' + basesports + '/accepted/' + translation + '/'
        textbase = basesports
    if basetravel in textbase:
        prefixrequest = '<|startoftext|>' + basetravel + '/accepted/' + translation + '/'
        textbase = basetravel
    if basescifi in textbase:
        prefixrequest = '<|startoftext|>' + basescifi + '/accepted/' + translation + '/'
        textbase = basescifi
    if baseask in textbase:
        prefixrequest = '<|startoftext|>' + baseask + '/accepted/' + translation + '/'
        textbase = baseask
# Answer generation
    text = onnx_model.generate(sess,run_name=sportsrun,temperature=float(0.2),length=700,prefix=prefixrequest,batch_size=1,top_p=0.9,top_k=40,truncate='<|endoftext|>',include_prefix=False,return_as_list=True)
    generate_count += 1
    if generate_count == 100:
        del onnx_model
        del onnx_modelbase
        onnx_model = GPT2ONNXModel('gpt2_fp16.onnx', 'gpt2-xl', device='cuda', verbose=False)
        onnx_modelbase = GPT2ONNXModel('gpt2base_fp16.onnx', 'gpt2-xl', device='cuda', verbose=False)
        generate_count = 0
    liststring = text.split('/')
    answereng = liststring[0]
    answer = answereng
    linkslist = liststring[1]
    if fromlang != 'en':
        answer = translate_text(fromlang, tolang, answer).replace('<span class=\"notranslate\">' , '<').replace('</span>' , '>').replace('<div>' , '').replace('</div>' , '')
    gc.collect()
    return UJSONResponse({'answer': answer, 'links': linkslist, 'base': textbase, 'question': translation, 'answereng': answereng.replace('<','').replace('>','')},headers=response_header)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))