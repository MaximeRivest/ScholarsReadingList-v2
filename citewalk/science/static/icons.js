'use strict';
// One emoji per OpenAlex subfield (hand-picked), so a topic can be guessed at a glance.
// Field emoji are the fallback for anything not listed.
const FIELD_ICONS = {
  'Agricultural and Biological Sciences': '🌾', 'Arts and Humanities': '🎭', 'Biochemistry, Genetics and Molecular Biology': '🧬',
  'Business, Management and Accounting': '💼', 'Chemical Engineering': '⚗️', 'Chemistry': '🧪', 'Computer Science': '💻',
  'Decision Sciences': '🎯', 'Dentistry': '🦷', 'Earth and Planetary Sciences': '🌍', 'Economics, Econometrics and Finance': '📈',
  'Energy': '⚡', 'Engineering': '⚙️', 'Environmental Science': '🌳', 'Health Professions': '🩺', 'Immunology and Microbiology': '🦠',
  'Materials Science': '🧱', 'Mathematics': '📐', 'Medicine': '⚕️', 'Neuroscience': '🧠', 'Nursing': '👩‍⚕️',
  'Pharmacology, Toxicology and Pharmaceutics': '💊', 'Physics and Astronomy': '⚛️', 'Psychology': '🧩', 'Social Sciences': '👥',
  'Veterinary': '🐾',
};
const SUBFIELD_ICONS = {
  // agriculture & biology
  'Agronomy and Crop Science': '🌽', 'Animal Science and Zoology': '🦓', 'Aquatic Science': '🐟',
  'Ecology, Evolution, Behavior and Systematics': '🦎', 'Food Science': '🍞', 'Forestry': '🌲',
  'General Agricultural and Biological Sciences': '🌾', 'Horticulture': '🌷', 'Insect Science': '🐞', 'Plant Science': '🌿',
  'Soil Science': '🪱',
  // arts & humanities
  'Archeology': '🏺', 'Classics': '🏛️', 'Conservation': '🖼️', 'General Arts and Humanities': '🎭', 'History': '📜',
  'History and Philosophy of Science': '🔭', 'Language and Linguistics': '🗣️', 'Literature and Literary Theory': '📚',
  'Museology': '🏛️', 'Music': '🎵', 'Philosophy': '🤔', 'Religious studies': '🕊️', 'Visual Arts and Performing Arts': '🎨',
  // biochemistry, genetics, molecular biology
  'Aging': '👴', 'Biochemistry': '🧪', 'Biophysics': '🔬', 'Biotechnology': '🧫', 'Cancer Research': '🎗️', 'Cell Biology': '🫧',
  'Clinical Biochemistry': '🩸', 'Developmental Biology': '🐣', 'Endocrinology': '🦋', 'Genetics': '🧬', 'Molecular Biology': '🧬',
  'Molecular Medicine': '💉', 'Structural Biology': '🌀',
  // business
  'Accounting': '🧾', 'Business and International Management': '🌐', 'Industrial relations': '🤝',
  'Management Information Systems': '🗄️', 'Management of Technology and Innovation': '💡', 'Marketing': '📣',
  'Organizational Behavior and Human Resource Management': '🧑‍💼', 'Strategy and Management': '♟️',
  'Tourism, Leisure and Hospitality Management': '🏖️',
  // chemical engineering
  'Bioengineering': '🧫', 'Catalysis': '🔥', 'Chemical Health and Safety': '☣️', 'Filtration and Separation': '🫗',
  'Fluid Flow and Transfer Processes': '🌊', 'Process Chemistry and Technology': '🏭',
  // chemistry
  'Analytical Chemistry': '⚖️', 'Electrochemistry': '🔋', 'Inorganic Chemistry': '💎', 'Organic Chemistry': '🔗',
  'Physical and Theoretical Chemistry': '⚛️', 'Spectroscopy': '🌈',
  // computer science
  'Artificial Intelligence': '🤖', 'Computational Theory and Mathematics': '🧮', 'Computer Graphics and Computer-Aided Design': '🖌️',
  'Computer Networks and Communications': '📡', 'Computer Science Applications': '💻', 'Computer Vision and Pattern Recognition': '👁️',
  'Hardware and Architecture': '🖥️', 'Human-Computer Interaction': '🖱️', 'Information Systems': '🗃️', 'Signal Processing': '〰️',
  'Software': '👨‍💻',
  // decision sciences
  'General Decision Sciences': '🎯', 'Information Systems and Management': '📇', 'Management Science and Operations Research': '🚚',
  'Statistics, Probability and Uncertainty': '🎲',
  // dentistry
  'General Dentistry': '🦷', 'Oral Surgery': '🦷', 'Orthodontics': '😁', 'Periodontics': '🪥',
  // earth & planetary
  'Atmospheric Science': '🌦️', 'Earth-Surface Processes': '🏔️', 'Geochemistry and Petrology': '🪨', 'Geology': '⛰️',
  'Geophysics': '🌋', 'Oceanography': '🌊', 'Paleontology': '🦖', 'Space and Planetary Science': '🪐',
  // economics
  'Economics and Econometrics': '📈', 'Finance': '💰', 'General Economics, Econometrics and Finance': '💹',
  // energy
  'Energy Engineering and Power Technology': '🔌', 'Fuel Technology': '⛽', 'General Energy': '⚡',
  'Nuclear Energy and Engineering': '☢️', 'Renewable Energy, Sustainability and the Environment': '☀️',
  // engineering
  'Aerospace Engineering': '✈️', 'Architecture': '🏛️', 'Automotive Engineering': '🚗', 'Biomedical Engineering': '🦾',
  'Building and Construction': '🏗️', 'Civil and Structural Engineering': '🌉', 'Computational Mechanics': '🧊',
  'Control and Systems Engineering': '🎛️', 'Electrical and Electronic Engineering': '🔌', 'General Engineering': '⚙️',
  'Industrial and Manufacturing Engineering': '🏭', 'Mechanical Engineering': '🔧', 'Mechanics of Materials': '🔩',
  'Media Technology': '📺', 'Ocean Engineering': '🚢', 'Safety, Risk, Reliability and Quality': '🦺',
  // environmental science
  'Ecological Modeling': '🗺️', 'Ecology': '🦋', 'Environmental Chemistry': '🧴', 'Environmental Engineering': '♻️',
  'Global and Planetary Change': '🌡️', 'Health, Toxicology and Mutagenesis': '☠️', 'Management, Monitoring, Policy and Law': '📋',
  'Nature and Landscape Conservation': '🏞️', 'Pollution': '🏭', 'Waste Management and Disposal': '🗑️',
  'Water Science and Technology': '💧',
  // health professions
  'Complementary and Manual Therapy': '💆', 'Emergency Medical Services': '🚑', 'General Health Professions': '🩺',
  'Health Information Management': '🗂️', 'Medical Laboratory Technology': '🧫', 'Medical Terminology': '📖',
  'Occupational Therapy': '🧶', 'Pharmacy': '💊', 'Physical Therapy, Sports Therapy and Rehabilitation': '🏃',
  'Radiological and Ultrasound Technology': '📷', 'Speech and Hearing': '👂',
  // immunology & microbiology
  'Applied Microbiology and Biotechnology': '🍺', 'Immunology': '🛡️', 'Microbiology': '🦠', 'Parasitology': '🪱', 'Virology': '🦠',
  // materials
  'Biomaterials': '🦴', 'Ceramics and Composites': '🏺', 'Electronic, Optical and Magnetic Materials': '🧲',
  'General Materials Science': '🧱', 'Materials Chemistry': '🧪', 'Metals and Alloys': '🔩', 'Polymers and Plastics': '🧴',
  'Surfaces, Coatings and Films': '🎞️',
  // mathematics
  'Algebra and Number Theory': '🔢', 'Applied Mathematics': '📐', 'Computational Mathematics': '🧮',
  'Discrete Mathematics and Combinatorics': '🔣', 'Geometry and Topology': '🍩', 'Mathematical Physics': '♾️',
  'Modeling and Simulation': '🖥️', 'Numerical Analysis': '📉', 'Statistics and Probability': '🎲', 'Theoretical Computer Science': '🧮',
  // medicine
  'Anatomy': '💀', 'Anesthesiology and Pain Medicine': '😴', 'Cardiology and Cardiovascular Medicine': '🫀',
  'Complementary and alternative medicine': '🌿', 'Critical Care and Intensive Care Medicine': '🏥', 'Dermatology': '🧴',
  'Emergency Medicine': '🚑', 'Endocrinology, Diabetes and Metabolism': '🍬', 'Epidemiology': '📊', 'Family Practice': '👨‍👩‍👧',
  'Gastroenterology': '🫃', 'Geriatrics and Gerontology': '👵', 'Health Informatics': '📱', 'Hematology': '🩸', 'Hepatology': '🍷',
  'Immunology and Allergy': '🤧', 'Infectious Diseases': '😷', 'Internal Medicine': '🩺', 'Nephrology': '🫘',
  'Obstetrics and Gynecology': '🤰', 'Oncology': '🎗️', 'Ophthalmology': '👁️', 'Orthopedics and Sports Medicine': '🦴',
  'Otorhinolaryngology': '👃', 'Pathology and Forensic Medicine': '🔬', 'Pediatrics, Perinatology and Child Health': '👶',
  'Physiology': '🫁', 'Psychiatry and Mental health': '🧠', 'Public Health, Environmental and Occupational Health': '🏥',
  'Pulmonary and Respiratory Medicine': '🫁', 'Radiology, Nuclear Medicine and Imaging': '☢️', 'Rehabilitation': '🦽',
  'Reproductive Medicine': '🥚', 'Rheumatology': '🦴', 'Surgery': '🔪', 'Transplantation': '🫀', 'Urology': '🚽',
  // neuroscience
  'Behavioral Neuroscience': '🐀', 'Biological Psychiatry': '💊', 'Cellular and Molecular Neuroscience': '🧠',
  'Cognitive Neuroscience': '💭', 'Developmental Neuroscience': '👶', 'Endocrine and Autonomic Systems': '💓', 'Neurology': '⚡',
  'Sensory Systems': '👀',
  // nursing
  'Issues, ethics and legal aspects': '⚖️', 'Leadership and Management': '🧭', 'Nutrition and Dietetics': '🥗',
  'Research and Theory': '📝',
  // pharmacology
  'Drug Discovery': '💊', 'Pharmaceutical Science': '💊', 'Pharmacology': '💊', 'Toxicology': '☠️',
  // physics
  'Acoustics and Ultrasonics': '🔊', 'Astronomy and Astrophysics': '🌌', 'Atomic and Molecular Physics, and Optics': '🔦',
  'Condensed Matter Physics': '🧊', 'Instrumentation': '🎚️', 'Nuclear and High Energy Physics': '💥', 'Radiation': '☢️',
  'Statistical and Nonlinear Physics': '🌀',
  // psychology
  'Applied Psychology': '🧑‍🏫', 'Clinical Psychology': '🛋️', 'Developmental and Educational Psychology': '🧒',
  'Experimental and Cognitive Psychology': '🧩', 'General Psychology': '🧠', 'Neuropsychology and Physiological Psychology': '🧠',
  'Social Psychology': '👥',
  // social sciences
  'Anthropology': '🗿', 'Communication': '💬', 'Cultural Studies': '🎎', 'Demography': '👪', 'Development': '🌱', 'Education': '🎓',
  'Gender Studies': '⚧️', 'General Social Sciences': '👥', 'Geography, Planning and Development': '🗺️', 'Health': '❤️',
  'Human Factors and Ergonomics': '🪑', 'Law': '⚖️', 'Library and Information Sciences': '📚', 'Life-span and Life-course Studies': '⏳',
  'Linguistics and Language': '🔤', 'Political Science and International Relations': '🗳️', 'Public Administration': '🏛️',
  'Safety Research': '⛑️', 'Sociology and Political Science': '🏘️', 'Transportation': '🚆', 'Urban Studies': '🏙️',
  // veterinary
  'Equine': '🐴', 'Small Animals': '🐕', 'Food Animals': '🐄', 'General Veterinary': '🐾',
};
const iconFor = (subfield, field) => SUBFIELD_ICONS[subfield] || FIELD_ICONS[field] || '🔬';
