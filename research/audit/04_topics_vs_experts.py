"""OpenAlex primary topic vs. the 200-article expert gold standard (Rivest et al. 2021, Science-Metrix taxonomy).

OpenAlex subfields are Scopus ASJC categories; Science-Metrix subfields are a different scheme, so a
hand-written equivalence map is used (one OpenAlex subfield -> the Science-Metrix subfields that mean
the same thing or are its direct children). A pair counts as a subfield match when the expert's modal
first choice is in that set; a 'lenient' match when any subfield any expert listed is in it.
Field level: both schemes mapped to a coarse 12-bucket layer written below.
"""
import json, collections
rows=[r for r in json.load(open('research/audit/topics_expert200.json')) if r['W']]
EQ={ # OpenAlex (ASJC) subfield -> equivalent Science-Metrix subfields
 'Molecular Biology':['Biochemistry & Molecular Biology'],
 'Atomic and Molecular Physics, and Optics':['Optics','Chemical Physics','General Physics'],
 'Astronomy and Astrophysics':['Astronomy & Astrophysics'],
 'Biomedical Engineering':['Biomedical Engineering'],
 'Materials Chemistry':['Materials'],
 'Nuclear and High Energy Physics':['Nuclear & Particles Physics'],
 'Surgery':['Surgery'],
 'Plant Science':['Plant Biology & Botany'],
 'Infectious Diseases':['Tropical Medicine','Virology','Microbiology'],
 'Epidemiology':['Epidemiology','Public Health'],
 'Physiology':['Physiology'],
 'Electrical and Electronic Engineering':['Electrical & Electronic Engineering'],
 'Cellular and Molecular Neuroscience':['Neurology & Neurosurgery'],
 'Genetics':['Genetics & Heredity'],
 'Cell Biology':['Biochemistry & Molecular Biology','Developmental Biology'],
 'Pathology and Forensic Medicine':['Pathology','Legal & Forensic Medicine'],
 'Reproductive Medicine':['Obstetrics & Reproductive Medicine'],
 'Public Health, Environmental and Occupational Health':['Public Health','Epidemiology'],
 'Immunology and Allergy':['Immunology','Allergy'],
 'Artificial Intelligence':['Artificial Intelligence & Image Processing'],
 'Statistics and Probability':['Statistics & Probability'],
 'Global and Planetary Change':['Environmental Sciences','Meteorology & Atmospheric Sciences'],
 'Immunology':['Immunology'],
 'Ecology':['Ecology'],
 'Organic Chemistry':['Organic Chemistry'],
 'Small Animals':['Veterinary Sciences'],
 'Electronic, Optical and Magnetic Materials':['Materials','Optoelectronics & Photonics'],
 'Parasitology':['Mycology & Parasitology'],
 'Endocrinology, Diabetes and Metabolism':['Endocrinology & Metabolism'],
 'Animal Science and Zoology':['Zoology','Dairy & Animal Science'],
 'Obstetrics and Gynecology':['Obstetrics & Reproductive Medicine'],
 'Ecology, Evolution, Behavior and Systematics':['Ecology','Evolutionary Biology'],
 'Surfaces, Coatings and Films':['Materials'],
 'Hematology':['Cardiovascular System & Hematology'],
 'Radiation':['Nuclear Medicine & Medical Imaging','Nuclear & Particles Physics'],
 'Soil Science':['Agronomy & Agriculture'],
 'Computational Mechanics':['Applied Mathematics','Mechanical Engineering & Transports'],
 'Condensed Matter Physics':['Applied Physics','General Physics'],
 'Spectroscopy':['Analytical Chemistry'],
 'Rheumatology':['Arthritis & Rheumatology'],
 'Microbiology':['Microbiology'],
 'Oncology':['Oncology & Carcinogenesis'],
 'Finance':['Finance'],
 'Nephrology':['Urology & Nephrology'],
 'Computer Vision and Pattern Recognition':['Artificial Intelligence & Image Processing'],
 'Acoustics and Ultrasonics':['Acoustics'],
 'Experimental and Cognitive Psychology':['Experimental Psychology'],
 'Cancer Research':['Oncology & Carcinogenesis'],
 'Biomaterials':['Materials','Biomedical Engineering'],
 'Aging':['Gerontology'],
 'Insect Science':['Entomology'],
 'Biochemistry':['Biochemistry & Molecular Biology'],
 'Oceanography':['Oceanography'],
 'Pediatrics, Perinatology and Child Health':['Pediatrics'],
 'Periodontics':['Dentistry'],
 'Neurology':['Neurology & Neurosurgery'],
}
# coarse buckets for a field-level comparison
OA_FIELD={'Materials Science':'phys-mat','Medicine':'clinical','Physics and Astronomy':'physics','Biochemistry, Genetics and Molecular Biology':'biomed',
 'Agricultural and Biological Sciences':'bio-agri','Engineering':'engineering','Chemistry':'chemistry','Veterinary':'bio-agri','Immunology and Microbiology':'biomed',
 'Mathematics':'math','Computer Science':'ict','Environmental Science':'earth-env','Neuroscience':'biomed','Dentistry':'clinical','Earth and Planetary Sciences':'earth-env',
 'Psychology':'psych','Economics, Econometrics and Finance':'social'}
SM_FIELD={'Enabling & Strategic Technologies':'phys-mat','Clinical Medicine':'clinical','Physics & Astronomy':'physics','Biomedical Research':'biomed','Biology':'bio-agri',
 'Agriculture, Fisheries & Forestry':'bio-agri','Engineering':'engineering','Chemistry':'chemistry','Mathematics & Statistics':'math','Information & Communication Technologies':'ict',
 'Earth & Environmental Sciences':'earth-env','Psychology & Cognitive Sciences':'psych','Public Health & Health Services':'clinical','Built Environment & Design':'engineering'}
SM_SUB_FIELD_OVERRIDE={ # Science-Metrix puts these under 'Enabling & Strategic Technologies' but they sit elsewhere in ASJC
 'Materials':{'phys-mat','chemistry','engineering','physics'},'Nanoscience & Nanotechnology':{'phys-mat','chemistry','physics','engineering'},
 'Energy':{'phys-mat','engineering','chemistry','earth-env'},'Bioinformatics':{'biomed','ict','math'},'Biotechnology':{'biomed','bio-agri','phys-mat'}}
out=[]; agg=collections.defaultdict(lambda: collections.Counter())
for r in rows:
    eq=EQ[r['oa_subfield']]
    any_=[s.strip() for s in r['expert_any'].split(';')]
    sub_ok = r['expert_subfield'] in eq
    len_ok = any(s in eq for s in any_)
    oaf=OA_FIELD[r['oa_field']]
    smf = SM_SUB_FIELD_OVERRIDE.get(r['expert_subfield'], {SM_FIELD[r['expert_field']]})
    field_ok = oaf in smf
    out.append({**r,'subfield_match':sub_ok,'any_expert_match':len_ok,'field_match':field_ok})
    c=agg[r['subset']]; c['n']+=1; c['sub']+=sub_ok; c['any']+=len_ok; c['field']+=field_ok
for s,c in agg.items():
    print(f"{s}: n={c['n']}  subfield = modal expert: {c['sub']/c['n']:.0%}   subfield listed by any expert: {c['any']/c['n']:.0%}   coarse field match: {c['field']/c['n']:.0%}")
n=sum(c['n'] for c in agg.values()); print('all:', n, f"sub {sum(c['sub'] for c in agg.values())/n:.0%} any {sum(c['any'] for c in agg.values())/n:.0%} field {sum(c['field'] for c in agg.values())/n:.0%}")
json.dump({'rows':out,'summary':{s:dict(c) for s,c in agg.items()},'equivalence_map':EQ},open('research/audit/results_topics_vs_experts.json','w'),indent=1)
print('\nfield-level misses:')
for r in out:
    if not r['field_match']: print(' ',r['subset'][:4], r['oa_subfield'],'<'+r['oa_field']+'>','| expert:',r['expert_subfield'],'|',r['title'][:70])
