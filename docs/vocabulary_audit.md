# Medical, Pharmacy, and Basic Healthcare Vocabulary Audit

## 1. Overview and Objectives
In accordance with **Phase 2 & Section 14 (Medical Safety)** of the project requirements:
- The initial domain focus is **Medical - Pharmacy and Basic Healthcare Communication**.
- We audited the actual VDzSL dataset (415 signs) to identify genuine healthcare, symptom, dosage, and counter-interaction signs.
- **Strict Scientific Rule**: We **never** relabel unrelated signs (e.g. we do not treat `طبيب أسنان` [Dentist] as general doctor `طبيب`, and we do not invent missing signs like `دواء` [medicine]).
- This system is designed solely for **communication support**, not diagnostic or prescribing decisions.

---

## 2. Verified Healthcare Vocabulary in VDzSL (32 MVP Classes Selected)

The following signs have been verified in the dataset, cross-referenced with the 3DZSignDB SiGML lexicon and the Algerian Ministry of Solidarity Dictionary (2017), and selected for the focused healthcare baseline:

| # | Arabic Sign | French Meaning | English Meaning | Category | Videos | Avatar / Cameras | Relevance & Function |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- | :--- |
| 1 | **حسنا** | D'accord / Bien | OK / Alright | Acknowledgment | 20 | 4 Avatars / 5 Cams | Acknowledging pharmacist instructions |
| 2 | **طعام** | Nourriture / Repas | Food / Meal | Administration Context | 20 | 4 Avatars / 5 Cams | Crucial instruction: take with meals or on an empty stomach |
| 3 | **ماء** | Eau | Water | Administration Context | 20 | 4 Avatars / 5 Cams | Standard instruction: take tablet with a full glass of water |
| 4 | **أذن** | Oreille | Ear | Anatomy | 20 | 4 Avatars / 5 Cams | Common infection location (ear drops, otitis, pain) |
| 5 | **بشرة** | Peau | Skin | Anatomy | 20 | 4 Avatars / 5 Cams | Dermatological issues, rashes, allergies, or topical creams |
| 6 | **جسم** | Corps | Body | Anatomy | 20 | 4 Avatars / 5 Cams | Used to describe generalized body aches or specify pain locations |
| 7 | **رأس** | Tête | Head | Anatomy | 20 | 4 Avatars / 5 Cams | Essential for reporting headache / migraine symptoms |
| 8 | **قلب** | Cœur | Heart | Anatomy | 20 | 4 Avatars / 5 Cams | Vital organ; key for cardiovascular medication communication |
| 9 | **يد** | Main | Hand | Anatomy | 20 | 4 Avatars / 5 Cams | Injury site, burns, or instructions on applying topical medication |
| 10 | **يساعد** | Aider | Help / Assist | Assistance | 20 | 4 Avatars / 5 Cams | Patient requesting assistance or pharmacist offering guidance |
| 11 | **سكر** | Sucre | Sugar | Condition / Dietary | 20 | 4 Avatars / 5 Cams | Critical for diabetes discussions, glucose monitoring, sugar-free syrups |
| 12 | **نعم** | Oui | Yes | Confirmation | 20 | 4 Avatars / 5 Cams | Confirming symptom, allergy, or understanding instructions |
| 13 | **شكرا** | Merci | Thank you | Courtesy | 20 | 4 Avatars / 5 Cams | Standard closing of pharmacy exchange |
| 14 | **لا** | Non | No | Denial / Negation | 20 | 4 Avatars / 5 Cams | Denying allergy, symptoms, or confirming no prior conditions |
| 15 | **1** | Un (1) | One (1) | Dosage Number | 20 | 4 Avatars / 5 Cams | 1 tablet, 1 time daily, 1 box |
| 16 | **2** | Deux (2) | Two (2) | Dosage Number | 20 | 4 Avatars / 5 Cams | 2 tablets, 2 times daily (morning and evening) |
| 17 | **3** | Trois (3) | Three (3) | Dosage Number | 20 | 4 Avatars / 5 Cams | 3 times daily (morning, noon, night) |
| 18 | **نصف** | Demi | Half | Dosage Quantity | 20 | 4 Avatars / 5 Cams | Common instruction: split tablet / take half a tablet |
| 19 | **الليل** | Nuit / Soir | Night / Evening | Dosage Schedule | 20 | 4 Avatars / 5 Cams | Bedtime or evening dosing instruction (e.g., sleep aids, statins) |
| 20 | **صباح** | Matin | Morning | Dosage Schedule | 20 | 4 Avatars / 5 Cams | Morning dosing instruction (e.g., take before breakfast) |
| 21 | **كل يوم** | Chaque jour / Quotidien | Every day / Daily | Dosage Schedule | 20 | 4 Avatars / 5 Cams | Primary frequency instruction for chronic or acute medication |
| 22 | **طبيب أسنان** | Dentiste | Dentist | Healthcare Professional | 20 | 4 Avatars / 5 Cams | Indicates dental issue or prescription originating from dentist |
| 23 | **كم** | Combien | How much? / What price? | Inquiry | 20 | 4 Avatars / 5 Cams | Asking for cost of medication or quantity to take |
| 24 | **إبرة** | Aiguille / Seringue | Needle / Syringe / Injection | Medical Device | 20 | 4 Avatars / 5 Cams | Common pharmacy purchase for insulin, injectable medications, or vaccinations |
| 25 | **خطر** | Danger | Danger / Warning / Hazard | Medical Safety | 20 | 4 Avatars / 5 Cams | Crucial warning for contraindications, toxic exposure, or adverse drug reactions |
| 26 | **مريض** | Malade | Sick / Patient | Medical Status | 20 | 4 Avatars / 5 Cams | Identifies that a person or family member is unwell |
| 27 | **كحول** | Alcool chirurgical / Antiseptique | Alcohol / Antiseptic | Medical Supply | 20 | 4 Avatars / 5 Cams | Standard antiseptic requested at pharmacy counters for wound cleaning |
| 28 | **ألم** | Douleur | Pain / Ache | Medical Symptom | 20 | 4 Avatars / 5 Cams | Essential for patient describing symptoms or pain level to pharmacist |
| 29 | **حرارة** | Chaleur / Fièvre | Fever / Heat | Medical Symptom | 20 | 4 Avatars / 5 Cams | Critical symptom indicating infection or fever requiring antipyretics |
| 30 | **من فضلك** | S'il vous plaît | Please | Politeness / Request | 20 | 4 Avatars / 5 Cams | Polite formula initiating requests at pharmacy counter |
| 31 | **الآن** | Maintenant / Immédiatement | Now / Immediately | Temporal Urgency | 20 | 4 Avatars / 5 Cams | Urgent dosing (take immediately on onset of acute symptoms) |
| 32 | **مال** | Argent | Money / Cash / Payment | Transaction | 20 | 4 Avatars / 5 Cams | Payment and pricing interaction at counter |

---

## 3. Unsupported / Missing Healthcare Concepts (Gaps in VDzSL)

The VDzSL dataset is an open-domain everyday gesture collection derived from 3DZSignDB and contains critical lexical gaps in healthcare terminology. The table below documents concepts that **cannot** be recognized by models trained on VDzSL:

| Missing Concept (English) | Arabic Term | French Term | Clinical Impact | Mitigation & Data Collection Plan |
| :--- | :--- | :--- | :--- | :--- |
| **Medicine / Medication / Drug** | دواء | Médicament | High. Patient cannot explicitly sign the noun 'medicine' | Record human signing data or import SiGML from HamNoSys dictionaries; patient uses 'ألم' (pain) or brand/prescription card |
| **Pharmacy / Chemist** | صيدلية | Pharmacie | Medium. Concept usually implicit when physically present in the pharmacy | Include in planned secondary human signing collection |
| **Hospital / Clinic** | مستشفى | Hôpital / Clinique | High. Referral or origin of prescription cannot be signed directly | Priority collection in human signing stage |
| **Doctor / General Physician** | طبيب | Médecin | High. Must not substitute 'طبيب أسنان' when referring to a general doctor | Explicit safety rule forbidding relabeling 'طبيب أسنان' as general 'طبيب' |
| **Prescription / Order** | وصفة طبية / أوردونونس | Ordonnance | High. Patient currently presents physical paper prescription | Physical handover of prescription bypasses need for sign in immediate MVP |
| **Pill / Tablet / Capsule** | حبة / قرص | Comprimé / Gélule | Medium. Can be partially referenced by numeral + administration with water | Collect dedicated gesture clips for tablet/syrup administration |
| **Allergy** | حساسية | Allergie | Critical safety issue. Pharmacist must verify allergies via written card/app | Never assume allergy absence. Require written confirmation |
| **Blood** | دم | Sang | Medium. Blood pressure, blood test, or bleeding cannot be signed directly | Priority candidate for clinical LSA expansion |

---

## 4. Medical Safety & Operational Protocol

> [!CAUTION]
> **Safety Critical Boundaries**:
> 1. **No Autonomous Clinical Decision Making**: Under no circumstances should model outputs be used to infer diseases, diagnose conditions, or suggest drug dosages.
> 2. **Mandatory Visual & Written Confirmation**: For safety-critical instructions (e.g., dosage numbers, antibiotic course length), the pharmacist must confirm the meaning in writing or via physical packaging.
> 3. **Uncertainty Rejection**: Any model prediction with probability below an established confidence threshold or with high entropy must trigger an explicit **"Uncertain / Recheck"** status rather than guessing a potentially hazardous medical action.
> 4. **No Semantic Invention**: The absence of words like 'دواء' (medicine) or 'حساسية' (allergy) means the system must never attempt to hallucinate these words from similar signs.

---

## 5. Human Signing & Vocabulary Expansion Plan

To extend the vocabulary from the synthetic 3D avatar subset to full clinical utility:
1. **Target Vocabulary**: The 8 missing high-impact concepts identified above plus 25 common pharmaceutical phrases.
2. **Qualified Signer Participation**: Recruitment of native/fluent Algerian Sign Language (LSA) signers in collaboration with deaf associations in Algeria.
3. **Informed Consent & Ethics**: Standardized protocol ensuring participants provide informed consent for research recording, with options for face blurring / pose-only feature extraction to protect personal privacy.
4. **Multi-Camera Real Recording**: Capture in varied illumination and realistic pharmacy counter backdrops.
