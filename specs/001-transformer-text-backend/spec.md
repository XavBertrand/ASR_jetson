# Feature Specification: ASR Transformer Text Backend Unification

**Feature Branch**: `001-transformer-text-backend`  
**Created**: 2026-02-25  
**Status**: Draft  
**Input**: User description: "Goal: Le pipeline ASR doit utiliser TransformerAnonymizer (src/asr_jetson/postprocessing/transformer_anonymizer.py) comme backend texte. Non-goals: Ne pas changer le comportement PDF/DOCX/XLSX. Acceptance: le pipeline ASR appelle la nouvelle intégration (pas d'appel direct à un autre anonymizer texte), test de non-régression prouvant que run_transformer_anonymization() est utilisé. For text anonymization, the system MUST call asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization and MUST call asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization and, in **nominal mode**, MUST NOT introduce a parallel text anonymizer (see FR-003). In degraded mode, regex-only fallback is explicitly permitted (see “Execution Modes”)."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Backend Texte Unique (Priority: P1)

En tant que mainteneur ASR, je veux que l’anonymisation texte du pipeline passe par un backend unique pour éviter les divergences de règles entre chemins d’exécution.

**Why this priority**: C’est la valeur centrale demandée: supprimer les chemins parallèles d’anonymisation texte.

**Independent Test**: Exécuter le pipeline ASR avec anonymisation texte activée et vérifier qu’un test de non-régression prouve l’appel du backend unique attendu.

**Acceptance Scenarios**:

1. **Given** un run pipeline ASR avec anonymisation texte activée, **When** le texte est anonymisé, **Then** le pipeline passe par l’intégration `run_transformer_anonymization()`.
2. **Given** un run pipeline ASR avec anonymisation texte activée, **When** les composants d’anonymisation sont résolus, **Then** aucun autre anonymizer texte parallèle n’est utilisé.

---

### User Story 2 - Régression Contrôlée (Priority: P2)

En tant qu’équipe qualité, je veux un test automatisé qui détecte toute réintroduction d’un backend texte alternatif pour sécuriser les évolutions futures.

**Why this priority**: Sans garde-fou automatisé, la contrainte peut être cassée silencieusement dans un refactor.

**Independent Test**: Lancer une suite de tests ciblée qui échoue si `run_transformer_anonymization()` n’est plus appelé depuis le pipeline texte.

**Acceptance Scenarios**:

1. **Given** une modification future du pipeline, **When** les tests de non-régression sont exécutés, **Then** ils échouent si l’appel au backend texte canonique disparaît.

---

### User Story 3 - Stabilité Multi-Formats Documentaires (Priority: P3)

En tant qu’utilisateur métier, je veux que le comportement PDF/DOCX/XLSX reste inchangé pendant cette évolution pour éviter des effets de bord sur les usages existants.

**Why this priority**: Le périmètre demandé exclut explicitement les changements de comportement documentaires.

**Independent Test**: Exécuter les tests existants PDF/DOCX/XLSX pertinents et vérifier l’absence de dérive comportementale.

**Acceptance Scenarios**:

1. **Given** les traitements PDF/DOCX/XLSX existants, **When** la feature est intégrée, **Then** les résultats restent compatibles avec le comportement actuel.

### Edge Cases

If the canonical backend is **unavailable at runtime** (ImportError / init failure), the pipeline MUST activate **regex-only fallback** and MUST emit the canonical warning (see Warning Contract).
- Un développeur introduit un nouvel anonymizer texte dans un autre module: les tests de non-régression doivent le détecter.
- L’anonymisation est désactivée dans le pipeline: aucun appel backend texte ne doit être déclenché.
- Une refactorisation modifie les imports sans changer l’intention: le test doit vérifier le chemin effectif d’appel, pas seulement la présence d’un import.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Le pipeline ASR MUST utiliser un backend texte unique pour l’anonymisation des transcriptions.
- **FR-002**: Pour l’anonymisation texte, le système MUST appeler `asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization`.
- **FR-003**: En **mode nominal**, le système MUST NOT introduire ni utiliser un anonymizer texte parallèle pour le même flux pipeline.
- **FR-004**: Le comportement actuel PDF/DOCX/XLSX MUST rester inchangé par cette feature.
- **FR-005**: Le système MUST fournir une preuve automatisée de non-régression montrant que `run_transformer_anonymization()` est utilisé par le pipeline texte.

### Interface & Data Contracts *(mandatory when data is exchanged between modules)*

- Le contrat d’intégration pipeline->backend texte doit définir un point d’entrée unique pour anonymiser le texte.
- Cette feature ne crée pas de nouveau contrat de données métier; elle remplace/normalise le chemin d’appel backend texte.
- Aucun changement de schéma de `Document`, `Span`, `Entity`, `Mapping`, `Report` n’est requis par cette feature.

### Compatibility & Migration *(mandatory)*

- **CM-001**: Aucun changement de format mapping/report n’est introduit par cette feature.
- **CM-002**: Les workflows existants du pipeline ASR restent compatibles sans migration utilisateur.
- **CM-003**: En **mode nominal**, toute tentative d’utilisation d’un backend texte alternatif (autre que `run_transformer_anonymization()`) MUST échouer explicitement via tests de garde.
  - Exception: en **mode dégradé**, le fallback **regex-only** est explicitement autorisé (constitution) lorsque `run_transformer_anonymization()` est indisponible au runtime (ImportError / init failure).

### Execution Modes

- **Clarification**: The regex-only fallback in degraded mode is **not** considered a “parallel anonymizer backend” for FR-003 purposes; it is an explicitly permitted degraded-mode safety mechanism required by the constitution.
- **Mode nominal**: le backend canonique `run_transformer_anonymization()` est disponible et utilisé.
- **Mode dégradé (fallback)**: `run_transformer_anonymization()` est indisponible au runtime (ImportError / init failure). Le pipeline active alors un fallback **regex-only** (bounded/offline) conformément à la constitution.

### Warning Contract (Fallback)

When regex-only fallback is activated, the pipeline MUST emit a warning with:
- `warning_code`: `NER_UNAVAILABLE_REGEX_FALLBACK`
- `warning_message`: `NER unavailable => regex-only fallback`
- `warning_level`: `WARNING`
- Exposure: warnings MUST be exposed in `TextAnonymizationResult.warnings` (list of Warning objects) and MAY be mirrored to telemetry.

### Failure Modes & Controlled Degradation

- Si le backend texte canonique est **indisponible au runtime** (ImportError / init failure), le pipeline MUST activer le fallback **regex-only** + WARNING (cf. section “Degraded Mode: NER Unavailable”).
- Si le backend texte canonique est disponible mais échoue pendant l’exécution (exception), le pipeline MUST échouer avec une erreur explicite, actionnable et **sanitized** (pas de texte sensible).

### Degraded Mode: NER Unavailable (Constitution MUST)

If the canonical text backend (TransformerAnonymizer / run_transformer_anonymization) is unavailable at runtime (ImportError or initialization failure), the pipeline MUST:
1) fall back to a regex-only anonymization mode (offline, bounded),
2) emit an explicit WARNING exposed in `TextAnonymizationResult.warnings` with:
   - warning_code: NER_UNAVAILABLE_REGEX_FALLBACK
   - warning_level: WARNING
   - warning_message: "NER unavailable => regex-only fallback"
3) remain deterministic within a case_id and isolated across different case_id values,
4) never leak sensitive input text in logs/errors.

Acceptance:
- unit + integration tests prove fallback path + warning semantics.

### Test & Fixture Requirements *(mandatory)*

- **TR-001**: Les modules pipeline et anonymisation texte touchés MUST avoir une couverture de tests automatisés.
- **TR-002**: Un test de non-régression MUST prouver que `run_transformer_anonymization()` est appelé dans le chemin pipeline texte.
- **TR-003**: Un test MUST vérifier qu’aucun anonymizer texte parallèle n’est appelé dans le même flux.
- **TR-004**: Les tests de non-régression PDF/DOCX/XLSX pertinents MUST rester au vert.
- **TR-005**: Les tests MUST échouer si le backend texte unique est remplacé sans mise à jour explicite du contrat.

### Security & Secrets *(mandatory)*

- **SEC-001**: Cette feature ne doit pas introduire de nouvelle source de secret.
- **SEC-002**: Les erreurs/logs liés au backend texte ne doivent pas exposer de contenu brut sensible.
- **SEC-003**: Aucun secret ne doit être ajouté en dur au code source.
- **SEC-004**: Les garanties actuelles de confidentialité des journaux doivent rester inchangées.

### Document Redaction Guarantees *(mandatory when file formats are processed)*

- **DR-001**: Le comportement de redaction PDF existant doit rester inchangé.
- **DR-002**: Le comportement DOCX existant doit rester inchangé.
- **DR-003**: Le comportement XLSX existant doit rester inchangé.

### Determinism & Scope *(mandatory when placeholders/identifiers are generated)*

- **DET-001**: Les comportements de déterminisme déjà en place doivent rester identiques pour un même cas d’entrée.
- **DET-002**: Le périmètre de cette feature est limité à l’unification du backend texte dans le pipeline ASR.

### Determinism and Cross-Case Isolation (Constitution MUST)

- Determinism: For the same input text and the same case_id, anonymized output MUST be identical across runs.
- Cross-case isolation: For the same input text but different case_id values, anonymized placeholders/mapping MUST differ (no cross-case collisions), ensuring case separation.

Acceptance:
- one integration test verifies same-case determinism,
- one integration test verifies cross-case non-collision.

### Performance Baseline Protocol

- Dataset/fixture: `tests/data/pipeline/text_backend/sample_transcript.txt` (or a fixed corpus under tests/data/perf/)
- Command: `uv run pytest -q tests/perf/test_transformer_text_backend_performance_regression.py`
- Runs: 7 runs, report median wall time
- Metric: median elapsed time (seconds)
- Acceptance: regression <= 10% vs stored baseline (same machine class)

### Performance & Resource Constraints *(mandatory)*

- Le temps de traitement de l’anonymisation texte ne doit pas se dégrader de plus de 10% sur le scénario de test de référence interne.
- Aucun nouveau traitement parallèle texte ne doit être ajouté.
- Les limites runtime existantes du pipeline restent applicables sans configuration supplémentaire.

### Assumptions

- Le backend texte `run_transformer_anonymization()` est le backend de référence validé par l’équipe produit/qualité.
- Les tests PDF/DOCX/XLSX existants sont suffisants pour valider le non-impact demandé.
- La feature ne modifie pas le périmètre fonctionnel documentaire, uniquement le backend texte pipeline.

### Key Entities *(include if feature involves data)*

- **ASR Pipeline Run**: Exécution d’un traitement audio comprenant transcription et éventuelle anonymisation texte.
- **Text Anonymization Backend**: Point d’entrée unique responsable de transformer une transcription en version anonymisée.
- **Regression Guard Test**: Test automatisé garantissant la persistance du backend texte canonique.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% des runs de test pipeline avec anonymisation texte passent par le backend texte canonique.
- **SC-002**: 100% des tests de garde échouent lorsque le backend texte canonique est volontairement contourné.
- **SC-003**: 100% des tests PDF/DOCX/XLSX ciblés restent passants après intégration.
- **SC-004**: Le temps médian de traitement texte sur le scénario de référence n’augmente pas de plus de 10%.
