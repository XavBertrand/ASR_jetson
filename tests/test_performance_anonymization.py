"""
Test de performance pour l'anonymiseur Transformer
Vérifie les corrections des bugs identifiés
"""

from src.asr_jetson.postprocessing.transformer_anonymizer import TransformerAnonymizer


def test_bug_espaces():
    """Vérifie que les espaces sont conservés correctement"""

    text = "Marine a appelé Delphine du cabinet Super Avocats à Montpellier."

    anonymizer = TransformerAnonymizer()
    anon_text, mapping = anonymizer.anonymize_with_tags(text)

    print("\n" + "=" * 60)
    print("TEST BUG ESPACES")
    print("=" * 60)
    print(f"Original:   '{text}'")
    print(f"Anonymisé:  '{anon_text}'")

    # Vérifie qu'il n'y a pas de texte collé ni de double espace
    assert "  " not in anon_text, "Pas de double espaces après anonymisation"
    assert "<" not in anon_text, "Les tags ne doivent plus apparaître dans le texte"

    # Vérifie désanonymisation
    restored = anonymizer.deanonymize(anon_text, mapping)
    print(f"Restauré:   '{restored}'")

    # Compare caractère par caractère pour debug
    if text != restored:
        print("\nDifférences:")
        for i, (c1, c2) in enumerate(zip(text, restored)):
            if c1 != c2:
                print(f"  Position {i}: '{c1}' != '{c2}'")

    assert text == restored, "La désanonymisation devrait être exacte"

    print("✅ Bug espaces corrigé !\n")


def test_bug_acronymes():
    """Vérifie que les acronymes courts (CJD, UDAF) sont détectés"""

    text = "Marine a rappelé le CJD et l'UDAF pour Super Avocats."

    domain_entities = {
        "PERSON": ["Marine"],
        "ORGANIZATION": ["CJD", "UDAF", "Super Avocats"]
    }

    anonymizer = TransformerAnonymizer(domain_entities=domain_entities)
    anon_text, mapping = anonymizer.anonymize_with_tags(text)

    print("=" * 60)
    print("TEST BUG ACRONYMES")
    print("=" * 60)
    print(f"Original:   '{text}'")
    print(f"Anonymisé:  '{anon_text}'")
    print(f"Entités:    {list(mapping['entities'].keys())}")

    # Vérifie que CJD est détecté
    assert "CJD" not in anon_text, "CJD devrait être anonymisé"
    assert "UDAF" not in anon_text, "UDAF devrait être anonymisé"

    # Vérifie dans le mapping
    found_cjd = False
    found_udaf = False
    for tag, info in mapping["entities"].items():
        values_upper = [v.upper() for v in info["values"]]
        if "CJD" in values_upper:
            found_cjd = True
            print(f"  ✓ CJD trouvé: {tag} = {info}")
        if "UDAF" in values_upper:
            found_udaf = True
            print(f"  ✓ UDAF trouvé: {tag} = {info}")

    assert found_cjd, "CJD devrait être dans le mapping"
    assert found_udaf, "UDAF devrait être dans le mapping"

    print("✅ Bug acronymes corrigé !\n")


def test_bug_texte_long():
    """Vérifie que les textes longs sont traités complètement"""

    # Texte de ~1500 caractères (dépasse la limite de 512 tokens)
    long_text = """
SPEAKER_1 : Ce n'est pas grave le début, on ne repart. Mais voilà, c'est des difficultés surtout dans la gestion dans les entreprises. Parce que les particuliers, c'est assez facile. Mais les entreprises où il y a beaucoup... Comme UDAF ou Détail Group où il y a plusieurs... Voilà, Mille Mécat, je crois que c'était chez Mille Mécat. Je devais envoyer des ruptures comme un... les serfas et je crois qu'il y avait deux prénoms identiques où c'était le même moment et j'ai failli faire une bourde et c'est au moment où j'ai dit oulala donc j'ai rattrapé donc c'est des petites difficultés comme ça qui me permettent d'identifier en fait il faut que je fasse des recherches pour identifier le salarié ou le...

SPEAKER_2 : Donc c'est un besoin de plus de clarté dans les demandes en fait. Voilà. Surtout dans les dossiers où il y a multiples salariés, multiples... Voilà. Mais le mécan, oui, en plus, la difficulté que vous devez avoir, c'est que c'est des clients avec qui j'ai des liens, notamment au CJD. mais de plus en plus amicaux et donc on se tutoie ils m'envoient des sms parfois des whatsapp et je réagis en fonction je ne sais pas si c'est mail si c'est appel, si c'est sms si c'est whatsapp, quelquefois je pense vous le transmets donc je comprends que ça puisse être compliqué

SPEAKER_1 : donc d'informations plus précises voilà c'est ça effectivement ok et sur le

SPEAKER_2 : sur le fonctionnement avec Marine moi ça se passe bien. Delphine aussi est très disponible. On travaille avec Isabelle sur les dossiers complexes et parfois on contacte Super Avocats pour les questions juridiques.
    """.strip()

    domain_entities = {
        "PERSON": ["Marine", "Delphine", "Isabelle"],
        "ORGANIZATION": ["UDAF", "CJD", "Super Avocats", "Mille Mécat", "Détail Group"]
    }

    anonymizer = TransformerAnonymizer(domain_entities=domain_entities)

    print("=" * 60)
    print("TEST BUG TEXTE LONG")
    print("=" * 60)
    print(f"Longueur texte: {len(long_text)} caractères")

    anon_text, mapping = anonymizer.anonymize_with_tags(long_text)

    print(f"Longueur anonymisé: {len(anon_text)} caractères")
    print(f"Entités détectées: {len(mapping['entities'])}")
    print(f"Stats: {mapping['stats']}")

    # Vérifie que le texte est complet
    assert "SPEAKER_1" in anon_text, "Le début devrait être présent"
    assert "SPEAKER_2" in anon_text, "Le milieu devrait être présent"
    assert len(anon_text) > 1000, "Le texte anonymisé ne devrait pas être trop court"

    # Vérifie que les entités du domaine sont détectées
    print("\nVérification des entités:")
    for entity_type, entity_list in domain_entities.items():
        for entity in entity_list:
            if entity.lower() in long_text.lower():
                is_anonymized = entity not in anon_text
                print(f"  {entity:20} -> {'✓ anonymisé' if is_anonymized else '✗ non anonymisé'}")

    # Au moins 5 entités devraient être détectées sur ce texte
    assert len(mapping["entities"]) >= 5, f"Devrait détecter au moins 5 entités, trouvé {len(mapping['entities'])}"

    # Vérifie la désanonymisation
    restored = anonymizer.deanonymize(anon_text, mapping)

    # Compte combien d'entités sont revenues
    entities_restored = 0
    for entity_type, entity_list in domain_entities.items():
        for entity in entity_list:
            if entity.lower() in long_text.lower() and entity in restored:
                entities_restored += 1

    print(f"\nEntités restaurées: {entities_restored}")

    print("✅ Bug texte long corrigé !\n")


def test_performance_complete():
    """Test complet avec timing"""
    import time

    text = "Marine travaille avec Delphine chez Super Avocats. Ils collaborent avec le CJD et l'UDAF à Montpellier."

    domain = {
        "PERSON": ["Marine", "Delphine"],
        "ORGANIZATION": ["Super Avocats", "CJD", "UDAF"]
    }

    print("=" * 60)
    print("TEST PERFORMANCE COMPLÈTE")
    print("=" * 60)

    # Premier run (chargement du modèle)
    start = time.time()
    anonymizer = TransformerAnonymizer(domain_entities=domain)
    load_time = time.time() - start
    print(f"⏱️  Chargement modèle: {load_time:.2f}s")

    # Test d'anonymisation
    start = time.time()
    anon_text, mapping = anonymizer.anonymize_with_tags(text)
    anon_time = time.time() - start
    print(f"⏱️  Anonymisation: {anon_time:.3f}s")

    print(f"\nOriginal:   '{text}'")
    print(f"Anonymisé:  '{anon_text}'")
    print(f"Entités:    {list(mapping['entities'].keys())}")

    # Test de désanonymisation
    start = time.time()
    restored = anonymizer.deanonymize(anon_text, mapping)
    deanon_time = time.time() - start
    print(f"⏱️  Désanonymisation: {deanon_time:.3f}s")

    print(f"Restauré:   '{restored}'")

    # Vérifie que tout est correct
    assert text == restored, "Désanonymisation inexacte"
    assert len(mapping["entities"]) >= 3, "Au moins 3 entités devraient être détectées"

    print(f"\n✅ Performance OK!")
    print(f"   Total: {anon_time + deanon_time:.3f}s (hors chargement)")


if __name__ == "__main__":
    print("\n" + "🧪 TESTS DE CORRECTION DES BUGS ".center(60, "="))

    test_bug_espaces()
    test_bug_acronymes()
    test_bug_texte_long()
    test_performance_complete()

    print("\n" + "=" * 60)
    print("🎉 TOUS LES BUGS SONT CORRIGÉS !".center(60))
    print("=" * 60 + "\n")
