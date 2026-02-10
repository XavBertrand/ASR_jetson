import pytest
from pathlib import Path
from unidecode import unidecode

from src.asr_jetson.postprocessing.transformer_anonymizer import (
    TransformerAnonymizer,
    run_transformer_anonymization
)


def test_transformer_anonymization_basic():
    """Test basique d'anonymisation"""

    text = "Françoise a appelé Micheline du cabinet Top Avocats à Montpellier."

    anonymizer = TransformerAnonymizer()
    anon_text, mapping = anonymizer.anonymize_with_tags(text)

    print(f"\n🔍 Test basique")
    print(f"Original: {text}")
    print(f"Anonymisé: {anon_text}")
    print(f"Mapping: {mapping['entities'].keys()}")

    # Vérifie que les noms sont anonymisés
    assert "Françoise" not in anon_text, "Françoise devrait être anonymisé"
    assert "Micheline" not in anon_text, "Micheline devrait être anonymisé"
    assert "<PERSON_" not in anon_text, "Les tags XML ne doivent plus être présents"

    # Vérifie la structure du mapping
    assert "entities" in mapping
    assert "reverse_map" in mapping
    assert "pseudonym_reverse_map" in mapping
    assert len(mapping["entities"]) > 0
    pseudonyms = list(mapping["pseudonym_reverse_map"].keys())
    assert pseudonyms, "La table des pseudonymes ne doit pas être vide"
    assert any(pseudo in anon_text for pseudo in pseudonyms), "Le texte doit contenir les pseudonymes"

    # Vérifie que les espaces sont conservés
    assert "  " not in anon_text, "Pas de double espaces"

    # Vérifie la désanonymisation exacte
    restored = anonymizer.deanonymize(anon_text, mapping)
    print(f"Restauré: {restored}")

    # La désanonymisation doit être exacte (ou presque, accepte variations mineures d'espaces)
    restored_norm = unidecode(restored).lower()
    assert "francoise" in restored_norm
    assert "micheline" in restored_norm
    assert "top avocats" in restored_norm

    print("✅ Test basique OK")


def test_transformer_with_domain_entities():
    """Test avec entités du domaine"""

    # Définis les entités connues de ton domaine (IMPORTANT: inclure les acronymes)
    domain_entities = {
        "PERSON": ["Françoise", "Micheline", "Gertrude"],
        "ORGANIZATION": ["Top Avocats", "CJD", "UDAF", "Mille Mécat"]
    }

    text = ("Françoise a rappelé le client du CJD pour le compte du cabinet Top Avocats. "
            "Micheline devait valider la rupture conventionnelle.")

    anonymizer = TransformerAnonymizer(domain_entities=domain_entities)
    anon_text, mapping = anonymizer.anonymize_with_tags(text)

    print(f"\n🔍 Test domaine")
    print(f"Original: {text}")
    print(f"Anonymisé: {anon_text}")
    print(f"Entités détectées: {list(mapping['entities'].keys())}")

    # Vérifie que toutes les entités du domaine sont détectées
    assert "Françoise" not in anon_text, "Françoise devrait être anonymisé"
    assert "Micheline" not in anon_text, "Micheline devrait être anonymisé"
    assert "Top Avocats" not in anon_text, "Top Avocats devrait être anonymisé"
    assert "CJD" not in anon_text, "CJD devrait être anonymisé"
    assert "<" not in anon_text, "Le texte anonymisé ne doit pas contenir de chevrons de tag"
    assert mapping.get("pseudonym_map"), "Le mapping doit exposer les pseudonymes"

    # Vérifie qu'Top Avocats est bien classé en ORGANIZATION
    found_org = False
    for tag, info in mapping["entities"].items():
        values_lower = [v.lower() for v in info["values"]]
        if "top avocats" in values_lower:
            assert info["label"] == "ORGANIZATION", f"Top Avocats devrait être ORG, pas {info['label']}"
            found_org = True
        if "cjd" in values_lower:
            assert info["label"] == "ORGANIZATION", f"CJD devrait être ORG, pas {info['label']}"

    assert found_org, "Top Avocats devrait être détecté comme ORGANIZATION"

    print("✅ Test domaine OK")


def test_transformer_on_realistic_long_text():
    """Test sur le texte long réaliste de transcription"""

    raw_text = """
SPEAKER_1 : D'accord, c'est pas grave, le début on n'aura pas. Mais voilà, c'est des difficultés surtout dans la gestion dans les entreprises, parce que les particuliers c'est assez facile, mais les entreprises où il y a beaucoup. . . Comme Udaf ou Detail Group où il y a plusieurs. . . Voilà, Migmeca, je crois que c'était chez Migmeca, je devais envoyer des ruptures conventionnelles. des serfas et je crois qu'il y avait deux prénoms identiques ou c'était le même moment et j'ai failli faire une bourde et c'est le moment où j'ai dit oulala donc j'ai rattrapé donc c'est des petites difficultés comme ça qui me permettent d'identifier en fait il faut que je fasse des recherches pour identifier
SPEAKER_2 : le salarié. Donc c'est un besoin de plus de clarté dans les demandes en fait. Voilà. Surtout dans les dossiers où il y a multiples salariés, multiples. . . Voilà. Oui mais en plus la difficulté que vous devez avoir c'est que j'ai. . . c'est des clients avec qui j'ai des liens notamment au CJD. mais de plus en plus amicaux et donc on se tutoie, ils m'envoient des sms, parfois des whatsapp et je réagis en fonction puis je sais pas si c'est si c'est mail, si c'est appel, si c'est sms, si c'est whatsapp, quelquefois j'y pense ou le transmets mais donc je comprends que ça puisse être compliqué.
SPEAKER_1 : donc d'informations plus précises. Voilà, c'est ça effectivement.
SPEAKER_2 : Et sur le fonctionnement avec Françoise?
SPEAKER_1 : Les informations circulent bien donc j'ai pas de difficultés, non, ça de ce côté là. on se. . . oui c'est ça les informations circulent très bien parce que voilà quand on fait notre point, comme quand on faisait avec vous en fait, c'était ça c'est à dire, lui ne s'occupe de. . . moi quand je ne sais pas je lui pose la question si le dossier que je suis en train de traiter, est-ce que c'est elle ou est-ce que c'est vous qui le traitez pour savoir. . . vers qui je dirige sinon je vous l'envoie à vous quand je ne peux pas poser la question mais sinon les informations circulent bien et moi en mon sens on s'organise on dialogue bien donc il n'y a pas de soucis de ce côté là où elle garde les infos Non, je n'ai pas perçu ça. Et le mode de télétravail non plus, ça ne vous pose pas de problème? Moi, ça ne me pose pas de problème. Sauf pour le téléphone, quand je dois passer d'un téléphone, quand je dois, par exemple, j'ai un appel et qu'il veut vous parler, là par contre, ça, je ne le maîtrise pas. Je ne le pratique pas assez souvent, donc je préfère le dire, « Écoutez, je les garde en attente, je vois si vous pouvez les prendre de suite au téléphone ou pas, et de leur dire qu'on va les rappeler avec. . . » C'est le seul hic, c'est ça, c'est avec le téléphone, mais vous passez les appels sur votre portable.
SPEAKER_2 : Il y a d'autres moyens de faire après, sinon il y a peut-être. . . qui peut vous faire une petite démonstration et vous mettez une petite note à côté.
SPEAKER_1 : C'est ça, j'avais déjà essayé, on avait fait des tests avec Marie, je m'en rappelle, donc c'était un petit peu, pas complexe, mais en fait il faudrait que je pratique régulièrement. Voilà.
SPEAKER_2 : Oui.
SPEAKER_1 : Bon, après vous ne prenez pas. Je prends rarement les appels, mais c'est vrai que vous ne prenez rarement les appels. Et sinon, je dis que je m'en renseigne dès que vous êtes dispo.
SPEAKER_2 : Je préfère ça. Je ne sais pas si les clients le prennent mal, dans ce cas-là, il faut me le dire, mais moi, je préfère ça parce que c'est le plus gros problème. que je rencontre moi c'est les interruptions ça me fait perdre un temps mais je pense que c'est pour vous pareil quand vous avez des appels quand on est sur quelque chose que ce soit simple ou pas simple d'ailleurs peu importe mais qu'on a une interruption on est obligé de revenir en arrière et quelquefois je ne sais même plus sur quel dossier, comme je passe de dossier en dossier, quelquefois j'ai un appel et je me dis sur quoi je vais travailler juste avant, je ne sais plus, et comme j'ai tous les dossiers ouverts, je ne sais plus où j'ai arrêté. Donc je préfère effectivement ne pas prendre les appels sauf exception, quand ça se justifie ou quand j'ai le moment. fasse comme vous faites là, ça me va bien.
SPEAKER_1 : Oui, parce qu'au début à un moment donné je m'étais posé la question est-ce qu'il serait pas bien d'avoir une plage horaire, mais en fait on ne sait pas quand les clients vont appeler. Donc c'est pareil pour moi je m'étais posé la question je m'étais dit est-ce que la semaine est-ce que là, le temps qu'on fasse points etc on privilégie et puis je me dis mais les gens m'appellent et s'ils appellent c'est qu'ils
SPEAKER_2 : ont envie d'avoir quelqu'un enfin d'avoir pas forcément une réponse mais un contact une voilà c'est ça je pense que le plus important en tout cas c'est comme ça que je vois les choses et c'est pour ça que pour moi c'est pour ça que je suis pas d'accord avec Gertrude pour pas mal de choses sur la gestion d'un cabinet, mais pour moi c'est essentiel que le client il est quelqu'un, même si ce n'est pas la bonne personne qui l'attendait, quelqu'un qui le rassure, qu'il est quelqu'un, un humain au téléphone et pas quelqu'un d'un scandale téléphonique extérieur qui pourrait parce que ça pourrait être si seul, mais il y en a qui font ça. ça mais la personne en fait elle connaît pas le dossier elle elle est juste là pour faire le standard et donc le fait d'un pour moi pour moi c'est essentiel d'offrir ça aux clients et et quand je parle de la valeur ajoutée du cabinet votre poste et vous parce que votre poste tenu par vous, pour moi, est un des piliers parce que, justement, déjà, vous êtes mon. . . Comme vous me permettez d'avoir un petit sas de décompression, voilà, donc vous prenez en. . . Vous êtes en première ligne, mais c'est le poste qui veut ça. C'est ça. Donc ça, pour moi, c'est vraiment tellement important. Pour moi, c'est une grosse valeur ajoutée. Mais aussi pour le client, d'avoir toujours quelqu'un et d'avoir quelqu'un qui n'est pas complètement déconnecté du dossier, qui peut peut-être lui donner quelques informations, tout ce qui est procédure, tout ce qui est. . . Il est en cours, le dossier est en cours. ou je fais passer le message, et ainsi de suite, pour moi, c'est quand même important. Même si le client, bien sûr, il veut m'avoir moi au téléphone, et il n'est pas content s'il ne m'a pas. OK, mais je ne peux pas être partout. Exactement, il ne peut pas se multiplier. Et puis les clients sont parfois très, très exigeants, et il faut aussi. . . Il faut aussi, et ça c'est ce que j'apprends un petit peu, il faut aussi maintenir une certaine distance, pas une distance, parce que moi les distances émotionnelles je n'en ai pas, mais une certaine distance au moins physique en disant non en fait je ne suis pas à votre service 24h. sur 24.
SPEAKER_1 : Excusez-moi, mais voilà, de dire, hop, je sonne et instantanément. . . Et ça,
SPEAKER_2 : il ne faut pas leur donner cette habitude-là, parce que. . .
SPEAKER_1 : Et puis après,
SPEAKER_2 : on ne fait pas du bon boulot parce qu'une fois qu'on est sous l'eau, les fois où j'ai cette sensation d'être complètement sous l'eau et de dire, c'est là, c'est là où je suis. on peut faire des bêtises et tout ça donc c'est pas. . . ok pas d'autres difficultés? de toute façon ce que j'aime aussi c'est que les difficultés qu'on rencontre au fil du temps vous les dites et je vous encourage à le faire à ne pas rester bloquée voilà moi sincèrement vous l'avez dit tout à l'heure Autant pour Françoise que pour vous j'ai une énorme confiance, j'ai fermé que j'irai. Donc tout ce que vous me direz, même si ça ne me fait pas plaisir peut-être, je n'ai pas de souvenirs comme ça, mais je prendrai pas mal.
SPEAKER_1 : Voilà, il n'y a pas de difficultés particulières.
SPEAKER_2 : Qu'est-ce qui donne du sens à votre travail aujourd'hui? Contribuer au développement du cabinet, la cohésion d'équipe, aider et se coudre à l'écrire. Est-ce que vous voulez rajouter des choses là-dessus? Moi, c'est bien ça. J'apprécie. Vous savez que la lecture du questionnaire. . . m'a fait un plaisir mais un plaisir au lance c'était pas trop d'histoire j'ai eu un plaisir à les lire autant celui de Françoise que le vôtre parce que ça là qu'est-ce qui donne du sens à votre travail contribuer au développement du cabinet sachez-le que c'est rare mais ça fait partie
SPEAKER_1 : Alors oui, parce qu'on s'implique, on n'a pas que travailler, on a le salaire, effectivement, voilà, ça en fait, mais pour moi, être dans une entreprise, le but, ce n'est pas de le couler, ce n'est pas de le faire stabiliser, c'est de le faire développer et justement d'être acteur. et pas passives, et se dire, je fais mon travail, et puis après, voilà. Non, non, non, pour moi, c'est ça.
SPEAKER_2 : C'est une vraie richesse pour le cabinet, pour moi, d'avoir des personnes qui contribuent au développement du cabinet, et donc à sa bonne santé, à son rayonnement. pour moi c'est une vraie richesse parce que c'est vraiment pas que ce soit dans, j'ai pas d'autres exemples de cabinet d'avocats à part celui d'Gertrude j'ai pas d'avis là-dessus mais dans d'autres entreprises alors peut-être que c'est le fait qu'on soit petit aussi peut-être mais dans d'autres entreprises où les gens font leur métier et puis là l'exemple typique c'est quand j'ai été bloqué sur les salaires votre réaction à toutes les deux moi c'est chouette quoi chouette d'abord compris qu'on est un petit peu tard et puis votre votre réaction plutôt bienveillante et compréhensible alors même que c'est une contrepartie, vous l'avez fait le travail donc la contrepartie elle doit y être et pas en retard parce qu'en plus on sait très bien que tout le monde est allongé à la même enseigne donc le salaire qui tombe parce qu'il doit tomber à certaines barrières. voilà voilà voilà donc mais ça c'est c'est prenez enfin il ya et les consciences que déjà que moi je suis extrêmement reconnaissante de ça mais en plus que c'est plutôt rare donc oui je me fais super plaisir ça revient à
SPEAKER_1 : faire de dire c'est pas on n'est pas là pour faire coller le gamin on est là pour donc après effectivement chacun a ses échéances, il y a une quinzaine d'années encore un chacun, 3-4 ans que j'ai finalement, mais quand on a des échéances plus importantes de toute façon le salaire il se mange pas tout en 5 jours et heureusement, donc voilà si on peut aider C'est des choses qu'on peut faciliter si ça peut aider, donc il n'y a pas de problème.
SPEAKER_2 : Le cabinet aujourd'hui, vous le voyez très bien, il est en bonne santé, il marche bien, il marche de mieux en mieux, on s'améliore sur plein de trucs. Moi je suis très très optimiste et tout ça, par contre c'est ce gros problème du trésorerie que je n'arrive pas à. . . Alors à Copernic c'est vraiment ce sur quoi je bosse, c'est vraiment le point d'entrée. Alors pas que trésorerie, parce que la trésorerie en fait, à mon sens, est un. . . symptôme de plusieurs problématiques. Des problématiques qui ne sont pas vraiment graves, mais qui demandent d'évolution, qui demandent de traitements différenciés. Et donc c'est le cœur du sujet. Alors je ne parle pas que de trésorerie, mais c'est le problème du cabinet, c'est ce qu'il y a actuellement. Et puis pendant longtemps, j'ai fait, et je ne regrette pas, absolument pas, mais j'ai fait le côté un peu perso, parce que c'était mon projet, parce que c'est mon bébé aussi, et puis par loyauté aussi. Oh là là, non, ça ne veut pas que je parle de Philippe, parce que sinon il va y avoir une émotion qui monte. On va arrêter là. C'est vraiment le sujet. Le besoin d'évolution, c'est sur ça. Et il continue sur les autres sujets, mais on est vraiment sur le bon endroit. Je pense qu'on est, oui, oui, oui. Et puis ça, le fait de voir. . . qu'on était en conseil contentieux, qu'on commençait à être à l'équilibre, je me disais ça y est, tout les efforts vont, voilà, ça paye. Cohésion d'équipe, je vois ce que vous voulez dire, en tout cas moi je le perçois comme que votre travail a du sens parce que vous. . . Vous pouvez participer à la cohésion des groupes, c'est comme ça que je l'entends. Tout à fait, tout à fait. Aider et soutenir les clients, ça c'est plus ce qu'on fait dans le cabinet quoi.
SPEAKER_1 : Voilà, c'est ça. Mais être là quand même, d'être à l'écoute, de pouvoir faire le travail que je peux moi de mon côté, mais enfin. . . Aider, soutenir, voilà, parce que ça fait partie aussi de mon poste, c'est-à-dire d'être là et d'être réactive aux demandes et aux sollicitations des clients.
SPEAKER_2 : Oui, puis surtout, votre poste, à mon sens, c'est principalement de l'écoute et de la présence. les particuliers que les entreprises mais encore ils ont besoin ils ont juste besoin qu'on les prenne en compte donc quelquefois on n'a pas de solution et le nombre de fois où en face de moi le dernier c'était en visioconférence avec l'ANARO je leur ai pas donné de bonnes nouvelles j'ai pas de solution pour vous Vous êtes dans une situation où vous ne pouvez rien faire malheureusement, mais rien que l'effet de les écouter, de les comprendre, déjà ça fait le boulot.
SPEAKER_1 : C'est ça.
SPEAKER_2 : Comment évaluez-vous votre débit d'épanouissement au cours de cette année? Vous m'avez dit épanouie. Est-ce qu'il y a des choses qui vous permettraient d'être encore épanouie?
SPEAKER_1 : plus épanouie non pas forcément non non non ça me convient très bien non non les
SPEAKER_2 : missions sont motivantes et stimulantes pour vous motivantes aussi parce que voilà ça fait partie il n'y a pas des tâches qui vous qui vous alors ça je vais pas vous dire je vais vous les enlever forcément mais des tâches qui vous. . . saoulent? J'ai pas trouvé d'autres mots. Non, non, je vois ce que vous dites.
SPEAKER_1 : Ce que vous voulez dire, non, pas forcément, non, parce que justement, ça fait. . . Non, non, non, il y a. . . Non, franchement, tout ce qui est comptabilité,
SPEAKER_2 : ça vous pèse pas. Vous aimez ça, vous? Oui, oui.
SPEAKER_1 : Voilà, le nouveau passage de Penny Lane, ça y est, c'est. . . ça commence à être ancré voilà après ça ne fait pas forcément gagner beaucoup de temps mais bon voilà ça c'est les relances mais bon les relances ça fait partie du travail parce que les gens voilà il faut alors pas forcément non mais pas forcément dans le sens je veux pas C'est surtout le fait d'être là à faire quelque part, de demander, d'être payée du travail quoi. C'est ça.
SPEAKER_2 : Mais en fait, ça fait partie. . . Mais c'est pour ça qu'à chaque fois que je travaillais avec Patrick d'ailleurs, je l'ai eu au téléphone pour lui expliquer. expliqué pourquoi je partais de Ménéos, mais on ne veut pas retravailler, à chaque fois il me met dessus en me disant, fais des trucs automatiques.
SPEAKER_1 : J'ai essayé de regarder sur Cléos, mais c'est pas forcément quelque chose de très, très. . . Après je me suis basée sur un peu celui que vous aviez fait pour, à un moment donné vous aviez fait Relance, un pays où. . . relancement d'honoraires, blèves, etc. , je crois. J'ai repris un peu votre trame, vous les faisiez malider, mais au moins ça permet. Et puis la signature, en votre nom, enfin, votre signature, par exemple, plutôt que la mienne. Donc, du coup, mais voilà. Après, pas forcément. . . ce type mais d'avoir un support qui me permet de voir si on va une trame ou et
SPEAKER_2 : des trames prête après que vous adaptez en fonction du client en fonction de la situation tout ça mais ça c'était c'était c'était pour ça que je l'avais fait il ya un point sur le ça me fait penser un truc en sort du sujet Le problème qui est né avec M. Moral, qui a mal pris que l'on lui facture, alors que c'était dans la convention d'honoraires, donc normalement ils sont capables de compter le nombre de conclusions, mais bon, je comprends qu'ils ne le font pas. Elle me faisait remarquer que le client, si on leur dit qu'on vient de recevoir des conclusions, on a déjà fait deux jeux de conclusions, donc tout ce qu'on va faire maintenant, c'est facturer en plus. Elle me dit que le problème, c'est que le client va nous dire si on a besoin de conclure pour répondre à cette question. faut étudier le dossier, donc il faut facturer et donc je me suis dit il va falloir que dans mes conventions d'honoraires je mette un point sur l'étude des conclusions adverses dans la limite de jeu de conclusion par exemple et un tarif pour l'étude des conclusions et un tarif pour la rédaction
SPEAKER_1 : c'est ce qu'on s'était posé la question Parce qu'on en avait parlé toutes les deux, c'est pour ça que j'avais fait remonter l'histoire de mon âge. Et on l'a fait, je crois que c'est pour madame Ebrard, justement, en disant voilà. Mais effectivement, dans la convention, pour que votre travail au moyen de l'analyse soit pris en compte, soit facturable, il faut effectivement. . .
SPEAKER_2 : Parce que le fait d'étudier les conclusions. . . et de dire au client qu'il n'y a pas lieu, comme on a fait pour l'ONU, il n'y a pas lieu à rédiger d'autres conclusions, on va le mettre dans notre plaidoirie pour que ce soit dit oralement parce que ça suffit, c'est une prestation. Parce qu'en fin de compte, le client, face à ses conclusions, se dit, alors il y en a certains, ils ne veulent rien. ils vont nous faire une tartine, mais il y en a d'autres qui ne savent pas s'il faut faire des conclusions ou pas. Et donc nous, ça nous oblige, comme disait Françoise, de toute manière dans la gestion du dossier, même s'ils me disent, je ne veux pas que vous étudiez les conclusions, je ne veux pas savoir s'il faut conclure de nouveau ou pas. Moi, dans mon dossier, il va falloir que je le fasse. Ben oui, à se froncer. Mais ça, ça me regarde. Et si je ne veux pas faire du travail de qualité, je ne veux pas faire du travail de qualité, ça me regarde. Mais il faut que les clients prennent conscience. Et ce qui est le plus difficile, c'est le fait qu'il n'y ait pas de livrable. C'est-à-dire. . . de choses concrètes, ça ne veut pas dire qu'il n'y a pas de valeur. Et au contraire, la valeur, elle est plus un peu invisible parce qu'en fait, le contrat de travail, c'est ce que j'explique à mes clients pros, c'est que le contrat de travail, le document, en fin de compte, il n'a que peu de valeur. C'est un document, effectivement, on en trouve des contrats de travail, on en trouve sur Internet, on en trouve partout. mais c'est pas ça qui a de valeur la valeur c'est ce qui est juste avant et après mais le document en lui-même
SPEAKER_1 : c'est pas important
SPEAKER_2 : bien sûr qu'on a des trames comme Gertrude qui me dit pour Géorgie est-ce que tu peux me faire le contrat de travail aujourd'hui et non je ne peux pas, je ne peux pas, ah bah de retransmettre moi une trame de CDD et puis je vais faire avec, mais j'ai pas de trame de CDD en fait, je ne travaille pas avec des trames moi, alors si j'ai une trame, bien évidemment, mais pour moi je ne peux pas lui transmettre ça, parce qu'en fait elle ne va pas savoir quoi mettre, et heureusement qu'après coup, bon elle m'a eu, parce qu'elle m'a transmis. contrat et j'ai tout repris dans la journée donc bref mais son contrat heureusement que je l'ai revu parce qu'elle confondait deux types de CDD enfin c'est pas le livrable il a peu de valeur et notre valeur à nous c'est ce qu'il y a dans notre tête et le problème c'est que le client ne le voit pas et quand j'ai des clients qui disent oui mais on a discuté pendant un quart d'heure oui on a discuté pendant un quart d'heure mais si t'avais pas eu un avocat en face t'aurais pas vu ce genre de discussion enfin je t'ai donné des informations, je t'ai donné des conseils je t'ai donné des trucs donc ça a de la valeur c'est ça discussion et le psy qui parle pas tu discutes si ça va de manière qu'est ce qui vous motive le plus dans votre travail aujourd'hui et c'est des clients atteindre des objectifs et le salaire. Alors ça, ça m'a interpellé parce que ma satisfaction, oui, je confirme, je suis extrêmement satisfaite. Et celle des clients, je comprends aussi le nombre de ceux qui expriment. C'est trop rare qu'ils l'expriment, mais quand ils l'expriment, c'est souvent très agréable. Attendre les objectifs fixés. Et moi, à chaque fois, j'ai une culpabilité, c'est que je me dis « fixe pas d'objectifs ». Et quand je dis ça, je me dis « ah, c'est quoi des objectifs Pierrette? ».
SPEAKER_1 : Alors les objectifs, c'est justement de pouvoir, au niveau de la facturation, j'en reviens toujours au même point. Oui, mais ça fait partie des choses. On avait été confrontés à des moments justement où les mémos, vous n'aviez pas de temps, etc. Donc tout ça, pour moi, ça me fait partie des objectifs de se dire, de faire les choses plus régulièrement, donc de faire rentrer de l'argent plus régulièrement, et des objectifs quelque part. si vous les avez fixés puisqu'il y a un chiffre d'affaires, alors j'ai plus le mental en tête de se dire je crois que c'est 18 ou 20 milles depuis que entre 18 et 20 milles, alors oui moi je ne peux pas l'atteindre cet objectif mais de pouvoir facturer régulièrement ça me permet pour moi Pour moi, je me dis qu'on va atteindre cet objectif et puis surtout en mettant maintenant les frais de dossier, en faisant les honoraires forfaitaires en un seul et pas en trois ou quatre fois. C'est du travail que je passe en moins de temps sur la facturation. du temps pour autre chose donc alors l'objectif en lui-même c'est pas comme je dis c'est pas moi qui vais pouvoir facturer non mais vous participez mais voilà j'ai participé donc en fait ça a aidé à atteindre ses objectifs
SPEAKER_2 : donc vous c'est le chiffre d'affaires mensuel voire annuel on est bien parti d'ailleurs cette année je crois oui Et le fait de faire en sorte, enfin oui, si, l'objectif c'est de faire en sorte qu'il y ait toujours un peu de trésorerie pour que l'argent rentre régulièrement.
SPEAKER_1 : Voilà, on n'attend pas la fin du mois qu'on fasse toutes les factures qui seront réglées dans 15 ou dans 20 jours. Alors que d'ici la fin du mois, il faudra avoir entré tant d'argent pour. . . palier ou aux charges fixes.
SPEAKER_2 : Et c'était l'objectif de Penny Lane, quand j'avais pris Penny Lane, ce qu'on m'avait vendu, c'est que vous alliez gagner du temps, et je me suis dit, ce temps-là, ça sera intéressant que Pierrette, elle fasse plus de suivi de ce que vous faites, de suivi de trésorerie. les échéances, c'est un vrai soulagement pour moi d'avoir ce tableau, même s'il est toujours dans mon parafeur, mais je sais qu'il est là, en fait, quelquefois je ne le traite pas, et quelquefois vos demandes s'attendent longtemps avant que j'y réponde, c'est vrai, mais je sais qu'il est là, et je sais que c'est génial. et quand je vois que le compte est un peu en fébrile, je peux voir ce qui va être prélevé juste après, au moment. Et c'est super important d'avoir ça. Donc, je ne sais plus pourquoi je disais ça. Oui, Pénilène. Et c'était vraiment pour que vous puissiez faire des relances, faire plus souvent des factures, plus régulièrement tout ça, et faire un suivi des échéances de la trésor et du suivi de l'atteinte de l'objectif et tout ça.
SPEAKER_1 : C'est ça. Et de ne pas travailler non plus sans être trop réglé. maintenant être vigilante et ça de ce côté là c'est alors c'est pas des objectifs vraiment mais ça fait partie des choses on travaille dans la condition que la compte soit versée ça fait partie c'est cette vigilance que
SPEAKER_2 : Françoise a aussi et que vous avez que moi je commence à avoir aussi et c'est ça qui à moi c'est ça qui nous fait vivre de manière plus facile en ce moment.
SPEAKER_1 : C'est ça, ça rentre, ça sort, ça fait un cycle qui ne nous laisse pas dans la panade pendant deux ou trois semaines peut-être, et puis vos apports qui sont quand même assez forts. l'objectif c'est de ne pas avoir
SPEAKER_2 : l'objectif c'est quand même je vais fêter mes 10 ans de solo pour moi je me dis maintenant c'est plus possible maintenant tu vas gagner ta vie parce que c'est bien joli de bosser, alors moi je me régale, je m'éclate dans ce que je fais et c'est extrêmement important. Je trouve des sources d'épanouissement, je m'amuse, j'apprends, j'évolue, donc tout ça c'est génial, mais à un moment donné j'ai la pression aussi. Des enfants qui vont dire bientôt, et puis en plus qui commencent à avoir des volontés de faire des études. À chaque fois, je fais « Ouh! » J'ai intérêt à bien gagner ma vie maintenant, même si sa vie contribue bien, bien sûr. Donc, en 10 ans, oui, en 10 ans, j'ai fait des efforts, j'ai pris sur moi, j'ai moins de sang. Aujourd'hui, maintenant, je choisis que. . . Après, effectivement, ça n'a jamais été mon objectif. Et on en a tous besoin. Et le salaire, ça c'est une question que je veux vous poser. Le salaire, il n'évolue pas de. . . depuis très longtemps, sauf les évolutions normales, de façon collective, moi c'est quelque chose qui me. . . je ne vais pas dire que ça me gêne, mais ça me questionne, parce que je pense qu'on n'a pas. . . et c'est là où je voudrais qu'on en discute, c'est que je pense qu'on n'a pas la même perception des choses. Le fait que votre coefficient ne bouge pas, moi, me dérange, personnellement. Parce que je me dis, il faut évoluer, il faut grandir, il faut prendre. . . Je ne vous gâche pas, c'est pareil. Mais je vous laisse finir, pardon. Et après, je vous. . . Et donc, je me dis, ma main, s'il faut. . . qu'il faudrait qu'on puisse faire évoluer ce vif, que ça contribue. Et quand je regarde la classification, puisque c'est ce que je dis à mes clients, donc je le fais pour moi, quand je regarde la classification, je me dis, ben oui, mais le step de plus, il faudrait aller au-delà de ce qui est. . . fait. Donc moi c'est ma perception. Le problème c'est qu'il faudrait qu'on fasse évoluer le poste. Il y a deux soucis. Le premier problème, c'est est-ce que le cabinet en a besoin? Et deuxième problème, est-ce que c'est une volonté de votre part? Je me dis, à un moment donné, je ne peux pas forcer les gens à vouloir faire d'autres choses, faire évoluer leur poste, modifier leur poste, tout ça. Par contre, l'évolution du salaire qui répond juste aux majorations, aux évolutions. . . de la convention collective, moi personnellement me dérange. J'aimerais, alors encore une fois, j'aimerais faire plus, mais c'est pas que. . . Pour moi, vous méritez que le salaire évolue un peu plus que seulement ça. Pour tout ce que vous avez dit et pour tout ce qu'on s'est dit jusqu'à présent. Votre implication, le fait que. . . que vous remettiez en question votre travail, le fait que vous soyez autonome, tous ces éléments-là. Quand je regarde de manière purement technique et purement juridique, ça correspond au poste, au coefficient, je n'ai pas de problématique là-dessus. mais quand je vois le Quand je vois ce que vous ce que vous faites pour moi vous méritez plus mais je suis bloqué par cette
SPEAKER_1 : la grille de
SPEAKER_2 : classification donc ça c'est mon point de vue non je vous laisse dire le vôtre
SPEAKER_1 : Oui, parce que c'est vrai que ça fait partie, j'ai bien regardé moi, j'ai regardé aussi la classification, c'est vrai que ça, le niveau supérieur, le coefficient supérieur, je ne sais plus, c'est pour déléguer, etc. Moi, mon poste ne nécessite pas de déléguer puisque je n'ai personne. aide, enfin voilà, et j'ai pas besoin enfin, j'ai pas besoin, non au niveau de mon travail ça me suffit, j'ai pas besoin d'avoir quelqu'un d'autre à côté mais effectivement la revalorisation de base qu'il y a eu là en mai ou juin elle est minime effectivement je trouve que voilà mais après c'est le minimum ah bah c'est le minimum voilà c'est à dire que après en tant que vous avez le choix quelque part peut-être de faire un peu bien sûr sans aller au dessus au coefficient ah mais de toute façon le coefficient c'est une chose la rémunération c'est une autre par rapport à l'ECOEF, il y a une revalorisation de 20 ou 30 euros à l'année. Mais après, on n'est pas limité à ça, c'est-à-dire qu'on peut faire, voilà. Je vais être franche avec vous, quand Françoise a été. . . Alors, ce n'est pas la jalousie, mais j'ai eu un peu de mal à comprendre pourquoi elle a mis une valorisation. un peu plus importantes que ce qui était le minimum conventionnel, je ne sais plus comment on appelle ça et moins avoir le minimum
SPEAKER_2 : il y a des informations que je dois vous donner voilà c'est pour ça
SPEAKER_1 : mais après c'est vrai que je me dis peut-être d'avoir une revalorisation un peu importante que le minimum de base. Après je sais que malheureusement ce qui en plus impacte sur la prime d'ancienneté et je sais la situation donc je ne suis pas non plus enfin comment dire je suis Je suis consciente de ça, donc du coup, si on augmente un peu, après les 15% de la prime d'ancienneté qui vient en plus, ça fait une charge supplémentaire, cotisations, etc. , etc. Après, je ne vous cache pas que niveau impôts aussi, moi, ça me ferait passer sur la tranche, là-dessus. Je n'ai pas regardé, mais je me dis peut-être que oui, aussi moi, de mon côté, je serais augmentée. en contrepartie je vais payer plus d'impôts oui après ça se calcule mais bon je tenais à vous à vous en parler de ce côté là, je n'ai pas voulu le marquer sur le questionnaire parce que pour moi je préfère en parler de vie de voix c'est plus facile d'en parler que de l'écrire
SPEAKER_2 : Il y a des informations qui vous manquent. L'augmentation Françoise, c'est qu'elle devrait, et la classification m'imposerait, de la passer au coefficient 350. Sauf que le cap entre 300 et 350 en termes de rémunération. . . step était énorme, vraiment énorme, c'était plus de 500 euros brut par mois et donc je l'ai reçu, je lui ai dit je sais que tu mérites ça, je sais que tu mérites les 350, sauf que la situation fait que je ne peux pas le faire et donc je lui ai proposé ce qu'elle a accepté. que je le fasse en deux temps donc je l'ai augmenté sans la faire passer au coefficient 350 je l'ai augmenté de la moitié du step en lui disant dès que je peux faire le tranche supplémentaire je le fais de toute façon dans un an je crois que c'est en janvier en janvier 2026 je ferai le step supplémentaire pour la passer à 350 parce qu'en fin de compte son son et puis il faut que je fasse juriste junior il faut que je fasse sauter mais voilà pourquoi elle a eu une augmentation et je m'étais dit au ce moment là en janvier bouge en janvier j'avais dit je fais ça pour Françoise parce que là elle voudrait me faire elle pourrait m'obliger à le payer le reste elle a compris aussi la situation et elle a accepté le fait que je fasse en deux temps mais n'empêche que donc ça c'était des des informations et je m'excuse de ne pas vous les avoir données parce que je ne me suis pas mis à votre place et j'aurais pu me dire, mais elle ne va pas comprendre, Pierrette, pourquoi j'augmente Françoise et que je ne lui fais rien pour elle. Et dans ma tête, deuxième information que je voulais vous donner, dans ma tête, quand j'ai fait ça, quand j'ai eu cette conversation avec Françoise et que j'ai décidé ça, Je me suis dit, pour Pierrette, je vais le faire plutôt une sorte de prime Macron ou quelque chose comme ça. Qui dit prime Macron, dit prime Macron aussi pour Françoise. Donc, j'étais un petit peu bloquée. Et puis, en plus, je voulais voir comment ça se passait l'année. J'ai toujours extrêmement peur. mettre dans une situation compliquée mais bon ça c'est des explications ça n'est aucunement des justifications
SPEAKER_1 : voilà ce que je c'est pas de la jalousie par rapport à tout je comprends chacune a son poste et mais je me suis posé la question je me suis pétillée
SPEAKER_2 : c'est pas une marque d'absence de reconnaissance de ma part c'est pas une marque en disant Françoise en fait plus que vous je suis plus contente de Françoise que vous je suis extrêmement satisfaite des deux sans aucune problématique maintenant effectivement et ce que j'entends de ce que vous m'avez dit c'est que c'est quelque chose qui est important pour vous certes vous comprenez la situation et que vous pouvez vous mettre un peu de frein sur ce point-là, mais c'est important pour moi de savoir que c'est quand même quelque chose qui marque, qui est important de votre côté. Donc ça, je le prends comme c'est, et je vous remercie d'être compréhensive et d'être consciente de la situation et pas brusque. regarder que maintenant est ce qu'il ya d'autres choses parce que Françoise par exemple m'avait dit ben ok pour ça mais par contre si on peut faire des tickets restos ça m'arrangerait parce que ça c'est quand même un gain pour moi qui moi jean qui mange tous les midis là ça me fait une aide plus le travail du train et puis le ticket resto, ça me permet de compenser la perte. Est-ce que vous, de votre côté, il n'y a pas des choses sur lesquelles moi je
SPEAKER_1 : peux aider? Non, parce que vous m'aviez proposé les tickets restos, donc non, ça ne m'intéresse pas et non, je ne vois pas. . . non il n'y a pas plus enfin si vous avez des idées non moi j'ai pas forcément c'était juste cette question que je voulais vous poser parce que j'ai attendu vous voyez mais non après c'est l'explication et c'est vrai que ça me paraît c'est tout à fait normal et ils sont co-efficients à la fin de voile, donc non, moi je suis à un niveau de poste quelque part, l'évolution comme j'ai marqué plus tard, l'évolution, moi prendre des responsabilités supplémentaires ou faire des formations supplémentaires, ça vous intéresse? sachant que derrière j'ai autre chose, enfin voilà. à titre perso, donc j'ai, non, pas d'évolution, plus de responsabilité, pas plus de formation, au moins qu'il y ait besoin d'une formation spécifique sur quelque chose, mais j'ai la réponse à ma question que je me suis posée quand j'ai eu ces infos, voilà. C'est exactement ça.
SPEAKER_2 : Et en plus, elle a toujours le modèle, l'exemple de sa copine Morgane chez les afromantaises, qui elle est depuis le début à 350. Et effectivement, elle ne doit pas comprendre parce qu'elle doit se dire, je fais autant qu'elle. je fais même peut-être certainement mieux, j'en sais rien, et je n'ai pas le même coefficient, je n'ai même pas la même chose. Je lui ai expliqué, je lui ai dit que là encore, ce n'est pas un manque de reconnaissance de ma part, ce n'est pas que je considère qu'elle fait moins bien le travail, mais là, je sais, alors on va en parler demain. ensemble, mais je sais qu'elle, elle peut évoluer sur d'autres types de tâches, notamment recevoir des clients, et ainsi de suite. Et il faut voir comment on peut faire, mais. . .
SPEAKER_1 : Voilà. Mais c'est bien, au contraire. Elle débute aussi dans sa. . . Donc ça fait partie des. . .
SPEAKER_2 : content de l'avoir évolué que ça ça fait partie des pour moi ça fait partie d'été des vrais moteurs de et c'est ce qui me fait tenir sur la longueur c'est c'est contribuer alors à faire vivre des familles certes tous les entrepreneurs ça en eux mais c'est aussi de voir l'évolution des personnes et Françoise elle est un exemple d'évolution et en plus ça me rend fière parce que quelquefois je me dis mais c'est toi qui apprend tout et c'est plutôt valorisant
SPEAKER_1 : c'est bien bien sûr
SPEAKER_2 : Alors, y a-t-il des valeurs que vous aimeriez voir davantage vis-à-vis de l'entreprise? Non. Il n'y a rien qui ne vous sent pas aligné avec vos propres valeurs? Il n'y a pas des choses que je vous fais faire qui sont contraires? Non, non. Ça c'est extrêmement important pour moi parce que pour moi c'est vraiment quelque chose que je ne supporterais pas pour moi. Oui.
SPEAKER_1 : Non, mais voilà, c'est pour ça que, et c'est pareil, je pense que ça ferait partie des choses, je vous demanderais, enfin si jamais il y avait quelque chose, je, voilà, ça fait partie des choses, je vous demanderais un entretien forcément, mais d'en discuter, d'échanger, donc non, non, non. La conciliation entre votre travail et votre vie personnelle, ça va aussi?
SPEAKER_2 : Oui. Je ne sais pas si vous vouliez dire d'autres choses, mais là, par exemple, vous avez. . . Il y a des aménagements qui pourraient améliorer, non, les aménagements ne sont pas conciliés, donc il n'y a pas de choses qui vous. . .
SPEAKER_1 : Non, le mercredi et le vendredi en télétravail, en fait, quelque part, ça me permet de m'organiser aussi au niveau perso, et comme vous m'avez demandé de finir en. . . peu plus tard et que je vous avais dit non pour moi voilà pour moi c'est pas possible puisqu'après j'ai une deuxième ou troisième journée comme tout le monde mais donc voilà pour moi c'était pas possible après après vu que vous essayez
SPEAKER_2 : de voir vos contraintes personnelles et familiales il faut aussi voir qu'il y a des potentiellement des choses des modifications d'horaires, des choses comme ça, qui pourraient être, c'est des aménagements, tout n'est pas possible parce que vous avez un poste très particulier, mais il y a des choses qui peuvent être faites.
SPEAKER_1 : Oui, mais bon, pour moi ça me convient. C'est vrai qu'à un moment donné je me suis posé la question de rester sur place et de rentrer un peu plus tôt, J'ai besoin quelque part un peu de. . . De sortir. C'est pas que je n'aime pas la présence, mais j'aime bien sortir de mon cadre de travail pour revenir après, même si je ne fais que courir. Mais bon, ça, c'est moi qui le veux. Parce que j'ai la possibilité de travailler, de rester là. Mais c'est le fait, justement, de pouvoir sortir. Donc, j'ai réflexion. je déchire ça, mais non, après je préfère rester sur mes horaires si ça vous convient à vous aussi non, non, non, c'est pas
SPEAKER_2 : mais c'est quand vous aviez la problématique avec votre maman, je me suis dit peut-être qu'elle aura besoin de faire autrement pour être vous seriez proche aidant oui, mais non, là j'ai fait
SPEAKER_1 : J'ai juste maintenant voulu ne pas faire ce choix parce que pour moi c'était trop contraignant et donc je suis très bien dans mon travail.
SPEAKER_2 : Comme ça, ça ne va pas bien.
SPEAKER_1 : Voilà.
SPEAKER_2 : Et le télétravail, je ne serais pas sincère si je vous disais que. . . Je préférais que vous soyez toujours là, mais l'expérience me dit que ça ne pose pas de difficultés.
SPEAKER_1 : Voilà ce que je disais, on en parlait tout à l'heure avec Françoise, ah oui parce qu'il va y avoir un changement, enfin pas un changement mais par rapport à Ambre qui descend. contre s'il faut switcher un autre jour je suis par contre le mercredi pour moi c'est inévitable de venir au moins le matin mais voilà après je suis ouverte si vous souhaitez switcher un autre jour admettons du vendredi si je vais vous
SPEAKER_2 : dire très clairement je ne suis pas là pour pallier aux décisions qu'ils avaient d'accord Elle a une façon de voir les choses qui est la sienne. Pour moi, elle est en train de se mettre une balle dans le pied. Elle avait quelqu'un de formé, elle avait quelqu'un de chouette. Et elle ne veut pas faire cet effort-là pour des raisons qui la regardent, et je peux le comprendre parce qu'on n'a pas tous la même perception. Mais ce n'est pas parce que Françoise m'a dit qu'il faudrait faire attention le mercredi. Mais en fait, Françoise n'était pas là pour faire le standard d'Gertrude. Elle prend le choix de ne pas avoir de standard ouvert, d'accueil téléphonique et physique ouvert tous les jours. C'est sa décision. je ne veux pas qu'elle. . . Donc il faudra forcément. . . Et pareil pour vous, c'est-à-dire que moi, mes salariés ne sont pas là pour faire leur standard. Et ça, ça sera vraiment un point d'honneur. Et si vous voyez quelque chose qui commence, enfin une dérive qui commence à revenir, il faut m'alerter de suite parce que ça, je ne veux pas. Je prends, je fais. . . choix d'avoir une base salariale et c'est un choix très personnel qui n'est pas compris, qui n'est compris par presque aucun des avocats que je contacte, mais je fais ce choix là, d'avoir une base salariale importante parce que c'est comme ça que je vais travailler et que j'y crois que ça va marcher, mais c'est pas pour prendre aussi des. . . les charges d'un autre cabinet, avec toute l'affection que j'ai pour Gertrude, qui a d'autres choix, qui préfère réduire la masse artérielle, certes, ok, d'accord, mais dans ce cas-là, il faut se débrouiller.
SPEAKER_1 : Ponctuellement, comme on dit, voilà, c'est ça.
SPEAKER_2 : et de voilà l'entraide mutuelle un problème et il ya le truc vous sentez vous reconnu pour votre travail et vos efforts de manière satisfaisante ça me rassure je suis content parce que c'est vraiment très important pour moi si vous excitez votre confiance je confirme, je suis confiante, en pleine confiance, et pleinement reconnaissante. Et c'est vrai que je vous l'ai déjà dit, et on en a parlé, si je pouvais reconnaître avec plein d'autres manières de faire, et notamment rémunération, je le ferais, mais on va y arriver. Et oui, c'est ça. J'essaye autant que je peux de faire des marques de connaissances autres que le salaire. J'espère que ça marche très bien. Je me sens considérée et respectée, heureusement. Je me sens valorisée, c'est très bien. Suffisamment de signes de connaissances, on va en parler. je sens que vous avez conscience en moi suffisamment responsabilisé on en a parlé en auto-dominance c'est ça alors voilà derrière cette question là c'était est-ce que je n'ai pas trop de responsabilité ah non non non non Parce que ça peut être ça aussi, je ne sais pas, on ne me délègue pas suffisamment, on ne me fait pas moins confiance, on ne me donne pas assez d'autonomie, ça c'est un point. Mais il y a aussi l'autre point où on met trop de charges et c'est pas. . .
SPEAKER_1 : Non, c'est vraiment. . . Non, non, là il n'y a rien.
SPEAKER_2 : Parfait. L'ambiance et la qualité des relations avec vos collègues est excellente. Il n'y a pas de difficulté de partante, ni avec un nain, ni avec un nain.
SPEAKER_1 : Non, non, non, non.
SPEAKER_2 : Ça c'est chouette. Qu'est-ce qui contribue le plus à une bonne ambiance selon vous? Le respect, la bonne entente, l'écoute, l'entrée, l'esprit d'Akima. Si on est honnête. Je crois que. . . Peut-être un ou deux mots près, je crois que vous avez eu exactement la même réponse que Françoise.
SPEAKER_1 : Ah bon? Et sans se concerter? Sans se concerter, c'est ça. D'accord. Donc ça, c'est chouette.
SPEAKER_2 : Ça aussi, c'est important pour moi. En fait, c'est comme quand on me dit, mais il va prendre des salariés comme ça, plein de temps, avec un seul avocat. Ils n'avaient pas encore dit, un seul avocat, avec tout du tout. que j'ai, une secrétaire à plein temps pour un seul avocat, ça ne marche pas. Je me dis, comment tu peux me sortir ça? Je ne dis rien à ce moment-là et je souris dans ma tête en me disant, mais si, ça marche. Après, vous verrez d'autres sur eux, ça c'est sûr. estimez-vous avoir les outils les moyens et l'environnement nécessaire pour bien travailler alors je suis satisfaite mais c'est là pour être mieux et là si vous parliez de la porte du lot essentiellement voilà en fait c'est pas qu'Ambre est trop
SPEAKER_1 : bruyante ça arrive elle a des écoutes bref elle est voilà Et effectivement, quand je le retrouve, oui, c'est bien.
SPEAKER_2 : Et c'est très agréable.
SPEAKER_1 : Mais après, c'est vrai qu'il y a des fois où je suis confrontée quand je suis au téléphone et qu'elle parle entre elle, ou même des fois quand elle parle un peu fort, ou même quand il y a du monde, elle n'appelle pas forcément elle. Au niveau du téléphone, je suis un peu embêtée. C'est surtout au téléphone, parce que quand je. . . C'est surtout au niveau de l'entente. Je suis obligée des fois de me mettre l'oreille comme ça, avec la personne au téléphone, pour pouvoir entendre bien ce qui se passe. C'est un petit détail, mais c'est vrai que voilà. Et cette porte, c'est vrai que des fois je me dis, alors je l'expliquerai, mais l'hiver, l'hiver aussi, donc déjà de pouvoir fermer quand justement il y a. . . trop de bruit et que moi j'ai le téléphone et ça me permet de bien écouter et en même temps pour le chauffage, le chauffage moi ça me, l'hiver ça m'a, la clim je ne l'ai pas mais l'hiver, le chauffage si je ferme, pas qu'importe, j'ai l'impression que tout s'en va. C'est dommage. Alors je ne veux pas m'isoler d'ambre qui est agréable comme tout, qui est comme on a dit, mais c'est surtout le fait de faire des économies au niveau du chauffage quand on est en hiver et d'avoir ce confort auditif quand je suis au téléphone.
SPEAKER_2 : J'ai bien pris note et de toute façon au moment où on est venu ici, quand j'avais visité les locaux, je m'étais dit cette porte, en plus elle n'est pas, cette porte il faut la changer, donc déjà j'avais l'idée de le faire, mais c'est vrai que je l'ai laissé tomber, je vais de moi, de moi, de moi, de moi, non non mais c'est. . . Quand je l'ai lu, j'ai dit, mais c'est vrai que cette porte, tu t'étais dit qu'il fallait la changer. Et c'est le moment où c'est le moment d'y réfléchir, parce que c'est juste changer une porte et mettre une porte qui ferme. Voilà, c'est ça. C'est ça qui est là et qui n'est pas du tout adapté. Je vais faire appel à papa. pour qu'il me donne des idées parce qu'en fait elle est très très lourde oui et donc je veux pas je veux pas me faire mal nous faire mal et lui faire faire moi aussi donc je voudrais qu'il me dise ce qu'il en pense et qu'est-ce que parce qu'il est plein de bonnes idées pour ce genre de choses et puis en plus ça le valorise un peu Il est à la retraite, donc toutes ces personnes-là disaient, je ne sers plus à rien. Mais si, tu sers. Tous les lundis, tu sers. Tous les mercredis, tu sers. Donc je vais lui demander ce qu'il lui ferait pour envisager ce qu'on fera peut-être pas dans l'avenir.
SPEAKER_1 : Oui, voilà, c'est ça. C'est juste, voilà. c'est ce que je rencontre en fait depuis deux. . . alors est-ce que peut-être moi aussi?
SPEAKER_2 : ça fait combien de temps que vous n'avez pas fait votre visite? bah non si vous l'avez mis il n'y a pas si longtemps que ça même si c'était auprès de médecins de travail
SPEAKER_1 : tout ce qui est oreilles et tout ça je ne sais plus mais bon moi c'est à moi aussi d'y faire attention et voilà je sais que l'âge en avance et du coup mais voilà c'est parce que j'ai un peu une baisse d'audition ou est-ce que c'est voilà, c'est ce petit souci
SPEAKER_2 : que je rencontre. Ça peut être aussi un problème de concentration vous avez plus de besoin de conditions de travail ce que vous disiez tout à l'heure c'est que pendant tout le temps vous avez bossé de manière un peu isolée et là maintenant vous êtes Vous avez un bureau séparé, mais vous avez une ambiance qui vient un peu vous vous polluer de manière très très. . . Oui, je pense qu'on m'embrasse trop. Est-ce que vous avez réfléchi, quand on a eu Oxycom, je vous l'avais demandé, le casque avec le micro, c'est quelque chose qu'on peut faire encore? Oui, mais ça, ça ne vous met pas à l'aise.
SPEAKER_1 : Non, c'est ça. C'est-à-dire que là, par contre, moi, je ne mets jamais les écouteurs. Parce que justement, j'aime bien avoir tous les prix environnants, etc. et ne pas être coupé, coupé, coupé. Donc non, ce n'est pas quelque chose qui m'intéresse. Par contre, effectivement, c'est de. . . de pouvoir entendre comme il faut et après moi au contraire le fait qu'elle passe parce que si je sais pas si vous en parliez mais moi au contraire le passage qu'elle quand elle passe en Guyana tout ça au contraire ça me permet voilà d'échanger de mots donc non je veux pas m'isoler complètement, c'est juste pouvoir avoir cette possibilité-là de fermer et de pouvoir entendre quand le téléphone, ou que la ligne est mauvaise, parce qu'il y a des fois les lignes ne sont pas très très bonnes, et du coup entre ça et le bruit ambiant à côté, j'ai un peu de mal, voilà, mais c'est vraiment un petit petit petit. . .
SPEAKER_2 : partie ce que vous disiez là c'est qu'on en a parlé je pense sur la partie sur les sur les conditions de travail et tout ça il n'y a pas d'autres besoins j'ai toujours dans l'idée de ce siège assis debout mais je le trouvais chaud j'ai failli tomber par terre je suis allée essayer je ne sais plus comment ça s'appelle Si vous n'avez pas d'exemple de fauteuil assis debout, vous savez, prenez le haut un peu élevé. Si, si, si, venez voir, on a un haut, je ne sais pas quoi, je ne sais plus comment on dit, une présentation. Et je vois un siège et je me dis, ah ouais, il est chouette en tout, super ergonomique et tout. Et je dis, donc il faut vraiment. . . combien, il m'a sorti un truc de 350 euros ou quelque chose, ça fait, ah oui, ah bah il peut être bien, alors il m'a expliqué que c'était quelque chose qui était un peu, c'est le siège qui oblige le dos de se rester, c'est pas un siège fixe, c'est un siège qui permet un petit peu comme les ballons, les choses comme ça qui obligent le corps, enfin le squelette à se porter je ne sais pas comment dire et donc elle me dit oui, c'est à nous de venir c'est machin, c'est tout mais n'empêche que sans aller sur un truc aussi aussi élevé, avoir que vous ayez quelque chose juste pour pouvoir vous appeler quand vous bossez debout, c'est quelque chose que je cherche.
SPEAKER_1 : D'accord, mais enfin voilà, moi par contre je m'en sers beaucoup, le fait de. . . Je vois ça, je suis contente. C'est quelque chose, alors ça ne résout pas mes soucis de douleur, mais ça c'est, ça vous êtes. . . Mais j'apprécie le fait de pouvoir alterner la position assise et la position. . .
SPEAKER_2 : Moi, l'objectif que j'ai, c'est que le travail ne vous porte pas encore plus de préjudice. Je ne peux rien faire pour votre état de santé, c'est sûr. Mais par contre, que le travail ne vous impacte pas négativement encore plus. C'est mon objectif, c'est ça?
SPEAKER_1 : Non, non, mais voilà. Je suis contente et merci pour l'investissement. Mais voilà, ça fait partie des choses. . .
SPEAKER_2 : Ce qui me faisait peur avec le. . . Et là, encore une fois, je l'ai vu avec ma propre perception, le fait de cette manivelle, j'ai dit, mais Pierrette, elle ne prendra jamais le temps de me lever et de me baisser. C'est ce que je vois, donc je suis contente.
SPEAKER_1 : Rassurez-vous de ce côté-là, mais non. Et après, je peste un peu, ça ne sert à rien d'avoir. C'est quand je cherche une place dans l'environnement. C'est pour dire dans les conditions.
SPEAKER_2 : Avec le parking de la gare qui est payant.
SPEAKER_1 : C'est ce qu'on m'a dit lorsque je disais mes cours. comment ça se fait qu'il y ait autant de monde maintenant, donc on m'a expliqué.
SPEAKER_2 : Et attendez, j'attends, j'ai qu'une crainte, c'est de voir fleurir des petits parfaîtres. On ne va pas s'inquiéter maintenant, on va voir, mais si ça arrive, on va pouvoir trouver un moyen parce que. . . la place là là qui est encore gratuite je trouve que c'est un miracle donc pour l'instant on en profite c'est ça ok après c'est tout ce qui concerne l'évolution puisque là c'était est-ce que vous avez le sentiment d'apprendre et de progresser dans votre rôle ou poste évoluer à rythme satisfaisant pas d'ennuis pas de non non Quelle évolution professionnelle envisagez-vous dans un an, trois ans et cinq ans? On en a parlé. Vous êtes bien maintenant? Si ça bouge, vous le dites. Oui, tout ce que je peux vous dire. S'il y a d'autres choses, voilà. Parce que ça peut évoluer d'abord. De quelle manière? Un cabinet petit, il n'y a pas d'évolution à l'usager. Là c'est pareil, c'est pas parce que vous m'avez dit ça que c'est figé et que si jamais vous avez une envie de. . . Parce que ça peut être aussi sur quelque chose qui peut vous aider dans vos autres engagements par ailleurs. qui peuvent bien sûr participer à l'adaptation à votre poste, mais aussi qui peuvent vous aider pour autre chose. Oui, il y a des. . . c'est possible.
SPEAKER_1 : D'accord.
SPEAKER_2 : Moi, je vais continuer à vous faire les propositions. Je sais que Françoise, elle, elle est friande de ça, donc je vais continuer. Mais il va y avoir. . . à cette année de forum, ce qu'on appelle les forums collaborateurs où on a des formations exprès pour nos salariés, je continuerai à vous les proposer. Oui. C'est pas parce que vous m'avez dit non jusque là que vous ne pouvez pas dire non. Que je suis fermée à tout. Oui, oui, bien sûr. Avez-vous le sentiment de savoir vers où le cabinet se dirige et quelle est sa vision pour l'avenir? Oui. Vous sentez-vous suffisamment prise en compte dans la réflexion autour du développement du cabinet? Oui, et j'ai envie de vous dire, je pense que ça va être de plus en plus, parce que Copernic, la formation que je fais, c'est vraiment essentiellement ça, c'est l'évolution de ma posture en tant que dirigeante. mais ça a un impact sur le cabinet et je ne vois pas faire ça toute seule. Et là encore, c'est pour ça que cette histoire d'entretien professionnel m'a interpellée, c'est qu'il faut que je fasse attention à ne pas tomber dans un excès à un autre. C'est-à-dire que je suis d'une. . . je suis plutôt à prendre des décisions, à aller vite, à faire comme moi je veux, et comme je connais cette manière-là de faire, et que j'ai du mal à l'assumer par certains côtés, un peu le côté bulldozer qui ne me plaît pas, Je compense en étant parfois trop dans la volonté d'être dans la participation, je ne prends pas de décision, je fais, j'évite. Et c'est là où j'ai percuté, c'est l'entretien professionnel qui m'a fait percuter ça, qui m'a fait dire, à un moment donné, tu as des responsabilités. Et prendre tes responsabilités, ce n'est pas forcément être un bulldozer. C'est prendre tes responsabilités parce que tu as tes responsabilités et te reposer sur Pierrette et Françoise, en l'occurrence. Ce n'est pas forcément bien parce qu'elles n'ont pas cette responsabilité-là. Elles ne doivent pas l'avoir. Et puis, elles ne veulent peut-être pas l'avoir. Donc, c'est là où je me suis dit, attention de ne pas tomber dans un accident. un autre. Certes, bulldozer ça peut faire mal aux gens, donc non, mais d'un autre côté, être dans le trop participatif, c'est nier mes responsabilités et c'est nier ma nature aussi, c'est pas ok. Donc n'hésitez pas quelquefois à me demander votre avis. Je sais bien, mais on va vous donner votre avis, mais c'est à vous. C'est à vous de le faire. C'est votre bébé, ça. N'hésitez pas, ça non plus, je le prendrai pas mal, parce que je sais que j'ai cette ambivalence, on va dire. Ces deux tendances qui peuvent être bloquantes, parce que c'est pareil. Parfois, je ne veux pas oser être dans la direction et c'est comme ça que je vois les choses. Parce que je me dis, si je fais ça, elles ne vont pas se sentir en capacité de donner leur avis, de donner leur point de vue, tout ça. En fin de compte, je pense qu'à force de vous le dire, que votre avis m'importe et que vous allez le faire ou moi. Voilà, ça fait partie des choses, oui. J'espère que vous avez toujours cette, et ça compte beaucoup pour moi, que je me donne l'impression d'avoir votre mot à dire, d'avoir votre liberté de parole. d'expression, de dire on fait comme ça, mais potentiellement moi je trouve qu'on pourrait faire comme ça, après la décision elle revient, c'est sûr et je ne peux pas vous. . . ça c'est quelque chose que je ne peux pas vous faire reporter et vous dire, c'est vous qui prenez la décision et vous prenez la responsabilité non, ça me démarque, je ne prends pas ça donc la décision c'est moi qui. . . l'apprend, mais moi je suis toujours très intéressant pour avoir vos propres inquiétudes. Les échanges, oui, bien sûr. Partagez-vous la vision et la direction du cabinet. Oui, ça va comme ça. Et là, c'était plus sur moi, les questions, les dernières questions, c'était essentiellement pour voir comment. . . Si mon fonctionnement ne pouvait pas avoir d'impact négatif, alors bien sûr, impact positif, ça me va bien aussi, mais si mon fonctionnement n'a pas d'impact négatif sur vous, de ce que j'ai compris, c'est qu'il y a des impacts positifs, ça c'est chouette, et je vous remercie de l'avoir dit. Mais par contre, le seul impact négatif que je verrais dans mon fonctionnement, c'est ce manque de réunion.
SPEAKER_1 : Oui, et comme je vous disais, c'est vrai qu'on le comble un peu en le faisant avec une Françoise, mais en physique, et après avec vous, on échange aussi. par le chat, parce que vous avez un emploi du temps aussi qui vous êtes très occupé. Donc, en étant toutes les deux Françoises, quelque part, ça permet quand même de bien dégrossir, de faire le point, notamment ce qu'on fait sur la semaine. pour savoir, moi, de mon côté, ce que je dois préparer, ce que je dois faire, si on me demande un renvoi, si je prépare le dépôt des conclusions, et qui sera prêt quand les conclusions sont valides, des choses comme ça. Donc, ça manque un peu quand on veut échanger sur plusieurs points, mais après, voilà, c'est pas. . . Vous voyez, on arrive. . .
SPEAKER_2 : Oui, mais je pense que ça vous met en difficulté parfois, notamment avec les clients, où vous n'avez pas forcément toutes les infos, et moi ça ne me va pas, alors on en avait parlé, même je vous avais dit, je fais un point avec Françoise, je fais un point avec Françoise beaucoup. plus fréquemment parce qu'on parle des dossiers, mais nous on devait faire un point notamment sur les factures, les gestes financiers, Penny Lane, les choses qui concernent votre poste et l'agenda, et ça je ne le fais pas, et ça ne me convient pas, maintenant dans le fonctionnement que j'ai depuis septembre où les enfants. . . prennent enfin depuis septembre depuis l'année ou les enfants ne pas besoin de moi pour aller à montauban ça me va très bien de rester chez moi le matin quand je peux donc il ya ça que je veux pas remettre en question parce que moi ça me permet de vraiment de me poser le matin et tout ça le fonctionnement l'organisation elle est pour moi bonne et n'empêche que ça ça me manque aussi d'avoir ces points plus régulièrement et je sais que c'est une demande de Françoise aussi d'avoir des points beaucoup plus régulier avec moi notamment alors à trois ou à deux ou voilà il faut peut-être à trois on n'a peut-être pas besoin de le faire tout le temps de manière hebdomadaire mais au moins qu'on ait un moment privilégié où vous pouvez faire avancer
SPEAKER_1 : les choses voilà, les choses que j'emmène de côté alors comme vous dites, c'est pas forcément une fois par semaine et une fois on va dire peut-être une fois tous les 15 jours une fois dans le mois en fonction de mais c'est pas quelque chose qui devrait comme on faisait au début tous les lundis après-midi voilà toutes les trois quelque part il y avait des choses que vous échangez toutes les deux et c'est normal sur les dossiers sur la vision enfin voilà il y a l'analyse etc donc ça c'est plus propre à Françoise et moi c'était plus des problèmes de des petites choses oui mais c'est pas des petites choses
SPEAKER_2 : mais c'est des petites choses qui en sont importantes et Et sur ce domaine-là, c'est des choses que je mets souvent de côté parce que je me laisse bouffer par les dossiers. Et ça fait partie de mon travail de dirigeante aussi, de gérante. Je suis avocat, oui, mais j'ai aussi la gérance du cabinet. Et donc toutes ces questions-là que je. . . que je ne traite pas parce que ça passe à côté, là, ce n'est pas OK non plus. Donc, il faut trouver un moyen. J'avoue que pour l'instant, je n'ai pas trouvé.
SPEAKER_1 : Moi, ce que je peux peut-être essayer aussi d'envisager, c'est-à-dire de voir le volume, enfin, pas le volume, mais les différents petits points que je pourrais, par exemple, arriver. . . besoin de votre accord ou d'en discuter avec vous etc c'est de se dire bon là peut-être que j'arrive quand même à 5 ou 6 ou 7 ou 8 points à voir ce serait peut-être bien de demander à Micheline quand est-ce qu'on pourrait se voir on peut se faire ça aussi c'est à dire se dire par exemple que ça vienne de vous dans la semaine il faudrait est-ce qu'il y aurait un moment ou un autre où on pourrait se caler pas forcément physique, physique c'est mieux, mais en fonction de vos possibilités, soit par téléphone, soit, ou faire peut-être un récap, de se dire, moi j'aurais besoin d'avoir un retour de votre part sur tel et tel point, comme ça peut-être ça peut déjà dégager un ou deux. . .
SPEAKER_2 : En fait, moi j'aimerais bien que ce soit quand même plus hebdomadaire, même si de temps en temps il n'y a pas beaucoup de choses.
SPEAKER_1 : Plutôt que de laisser cumuler.
SPEAKER_2 : Parce que ce qui est cumulé après c'est de la charge mentale et c'est pas bon. Mais quitte à ce que je fasse. . . l'investissement de webcam même si on est chez vous ou chez moi qu'on puisse faire ce point au moins en visio
SPEAKER_1 : avoir à réfléchir mais c'est vrai que ça pourrait être peut-être de ne pas laisser trop
SPEAKER_2 : ou alors que le lundi après-midi je reprends Je reprends mon fonctionnement d'avant, où je venais quand même travailler, et là, c'est vrai que mon lundi, c'est. . .
SPEAKER_1 : Mais non, mais parce que vous avez. . . Voilà, donc, comme je dis, c'est pas forcément hebdo, pas forcément toutes les semaines, il y a des semaines où j'aurais pas forcé. Je veux pas. . . C'est pas de ne pas avoir besoin de vous, mais c'est que j'ai pas de forcément. . . C'est pas de problème de ce côté-là. Pas forcément des choses qui attendent.
SPEAKER_2 : Oui, mais il vaut mieux que vous, dans ce cas-là, il vaut mieux qu'en début de semaine, donc lundi matin, vous fassiez le point sur des choses qu'il faut voir et me dire, là, par exemple, j'ai quelques petites choses à voir avec vous. Et donc, dans ce cas-là, je viens l'après-midi. parce que c'est lundi lundi soir je suis au tennis donc le fait d'être là c'est pas gênant c'est juste par par confort qu'en plus cet hiver je vais avoir froid à la maison donc j'ai pas voulu refaire la cheminée pour ça donc peut-être que je reviendrai au cabinet donc je ne vais pas vous remettre en question l'organisation parce que je comprends mais il faut remettre en question et parce que parce que c'est pas moi ça me convient pas mais autant autant que vous si lundi matin quand vous faites votre point vous voyez qu'il ya des choses à voir que vous me disiez et comme ça on se fixe un temps l'après midi moi je sais que il ya une raison de plus d'aller au cabinet voilà et si par contre le lundi matin vous voyez que vous n'avez pas vraiment besoin ou c'est juste une question et puis on la traite comme ça soit par le chat, soit par le téléphone pas de soucis mais effectivement que tous les lundis matins vous me fassiez un petit en gros que vous exprimiez le besoin de me voir ou pas c'est ça et on le fait lundi après midi ou parce que lundi après midi j'ai pris votre chose comme hier où j'étais sur la route ou albi voir comment comment dans la semaine on peut se trouver un point mais ça aussi ça fait partie des choses que je bosse à copernic C'est cette ambivalence entre avocat, j'ai tendance à me consacrer sur les dossiers et à en oublier, alors il y a du mieux, beaucoup mieux d'ailleurs, mais à en oublier tout ce qui est fonctionnaire dirigeante, parce que c'est moins facile pour moi. naturel et maintenant les dossiers c'est plus plus fluide il est midi 10 ceci j'ai pas vu le temps passer de toute façon j'ai juste un point important j'ai je vous préviens au niveau du cjd cette année c'est de nouveau encore un petit peu un petit peu chargé j'y trouve beaucoup beaucoup d'intérêt donc ça me nourrit beaucoup donc c'est pour ça que j'accepte mais Et c'est en train de se faire l'art, donc je vous le préviens. Je suis éligible pour le poste de présidence de la section Montauban. Et des retours que j'ai, en fait le fonctionnement ça se passe comme ça, c'est que les personnes qui sont éligibles vont être cooptées par les membres du CHD. On n'est pas candidat, c'est-à-dire qu'il y a des membres. . . Ces idées sont questionnées et ils désignent des personnes qu'ils voudraient voir au poste de président. Ils désignent trois noms. Et sur ces trois noms, on demande si on se lance ou on ne se lance pas. Et après, s'il y en a un, deux ou trois, on fait une élection en bonne et due forme. Là, on est dans la période de cooptation. C'est-à-dire que cette semaine, les JD, les membres du centre du CJD, cooptent, c'est-à-dire donnent des noms. De ce que j'ai cru comprendre, je vais faire partie des cooptés, quasiment certains. Donc en amont, je me suis posé beaucoup de questions, et savoir ce que je prenais, ce que je ne prenais pas, est-ce que ça correspondait à ce que je voulais. parce que je ne voulais pas être ce que je me connais. Je sais que si je suis cooptée et que je n'ai pas réfléchi, je me sentirais obligée de le faire. Ce que je ne voulais pas, je voulais que ce soit une décision que je prends moi et pour moi et parce que j'y vois un intérêt et parce que ça contribue à mon évolution, tout ça, pas de problème. Donc j'ai réfléchi et je sais. . . c'est que si je suis cooptée, je me présenterais ce qui a un impact. Alors, pour moi, ça n'aura pas un impact plus que l'impact du CJD actuellement. C'est-à-dire que ce que je fais actuellement, c'est-à-dire animer des formations, j'ai une responsabilité au niveau d'une des valeurs. si je dis ce qu'est la solidarité, tout ça, ça va partir si je suis vraiment élue. Mais ça sera remplacé par mes charges de présidence. Je vous le dis parce que comme j'ai cette volonté-là et que si je suis cooptée, je me présente, et si je suis élue, je le prendrai, que vous sachiez que. . . que je vais accepter certainement cette position si elle se présente. Si elle ne se présente pas, ça ne pose pas de problème. Je suis plus un peu dans le destin, c'est-à-dire que si ça se présente, c'est que c'est fait pour, si ça ne se présente pas, c'est que ce n'était pas fait pour. Donc je suis très à l'aise avec ça, mais c'est une des raisons. laquelle je faisais j'ai mis cet entretien professionnel maintenant parce que dans un temps ma réflexion je me suis dit je ne prends pas si jamais je sens pas que c'est pas stable d'accord c'est à dire que ma principale crainte c'est de mettre en péril l'équilibre du cabinet. Mettre en équilibre personnel, je le vois par ailleurs, c'est réglé, enfin c'est réglé, c'est discuté, mais mettre en péril la stabilité et l'équilibre qu'on commence à peine à obtenir, c'est niette. Donc si je sentais, au regard de votre entretien et au regard de Chine Françoise, que c'est. . . forcément fiable non pas parce que vous n'êtes pas fiable mais parce que vous avez d'autres envies d'autres notamment marit je veux pouvoir m'appuyer sur ça et pas mettre en péril ce qui veut c'est pour ça que j'ai fait cet entretien professionnel maintenant pour que si jamais je suis coopté je sache en toute connaissance de cause dire oui ou non J'y vais ou j'y vais plus? D'accord. Ça c'est plus de l'information et de la transparence que je tenais à vous donner parce que je sens bien que, en tout cas coopté, je pense que je vais l'être. Après élu, on verra bien, mais voilà. D'accord. on verra si c'est une bonne chose ou pas, je sais pas on verra
    """.strip()

    # Entités connues du contexte
    domain_entities = {
        "PERSON": ["Françoise", "Micheline", "Gertrude"],
        "ORGANIZATION": ["UDAF", "Détail Group", "Mille Mécat", "CJD", "Top Avocats"]
    }

    anonymizer = TransformerAnonymizer(domain_entities=domain_entities)

    print(f"\n🔍 Test texte long ({len(raw_text)} caractères)")

    anon_text, mapping = anonymizer.anonymize_with_tags(raw_text)

    print(f"Entités détectées: {len(mapping['entities'])}")
    print(f"Stats: {mapping['stats']}")
    print(f"Échantillon anonymisé (300 premiers chars): {anon_text[:300]}...")

    # Vérifie que SPEAKER_ n'est pas anonymisé
    assert "SPEAKER_1 :" in anon_text, "SPEAKER_1 devrait être préservé"
    assert "SPEAKER_2 :" in anon_text, "SPEAKER_2 devrait être préservé"

    # Vérifie qu'au moins quelques entités sont détectées
    assert len(mapping["entities"]) > 0, "Au moins quelques entités devraient être détectées"
    assert mapping.get("pseudonym_map"), "Les pseudonymes doivent être renseignés"
    assert "<" not in anon_text, "Le texte anonymisé ne doit pas contenir de tags XML"
    assert any(pseudo in anon_text for pseudo in mapping["pseudonym_map"].values()), "Le texte doit contenir les pseudonymes"

    # Vérifie que le texte est complet (pas tronqué)
    # On vérifie que les derniers mots du texte original sont présents ou anonymisés
    assert len(anon_text) > len(raw_text) * 0.8, "Le texte anonymisé ne devrait pas être beaucoup plus court"

    # Test de désanonymisation
    deanon_text = anonymizer.deanonymize(anon_text, mapping)

    # Vérifie que les entités du domaine sont revenues
    entities_found = 0
    for person in domain_entities["PERSON"]:
        if person.lower() in raw_text.lower() and person in deanon_text:
            entities_found += 1

    for org in domain_entities["ORGANIZATION"]:
        if org.lower() in raw_text.lower() and org in deanon_text:
            entities_found += 1

    print(f"Entités du domaine retrouvées après désanonymisation: {entities_found}")

    # 🧪 Cas supplémentaire : stabilité sur les homophones/homographes
    homophone_context = (
        "SPEAKER_3 : Françoise confirme la réunion même si tout le monde est fatigué.\n"
        "SPEAKER_3 : Françoise relance le client malgré la faute de casse.\n"
        "SPEAKER_4 : On parle ensuite de la vie Françoise fragile près du port.\n"
    )

    def _nth_index(text: str, substring: str, occurrence: int = 1) -> int:
        idx = -1
        for _ in range(occurrence):
            idx = text.index(substring, idx + 1)
        return idx

    Françoise_upper_start = _nth_index(homophone_context, "Françoise")
    Françoise_lower_start = _nth_index(homophone_context, "Françoise")
    Françoise_sea_start = _nth_index(homophone_context, "Françoise", occurrence=2)

    homophone_entities = [
        {
            "start": Françoise_upper_start,
            "end": Françoise_upper_start + len("Françoise"),
            "entity_type": "PERSON",
            "score": 0.98,
            "source": "test",
        },
        {
            "start": Françoise_lower_start,
            "end": Françoise_lower_start + len("Françoise"),
            "entity_type": "PERSON",
            "score": 0.9,
            "source": "test",
        },
        {
            "start": Françoise_sea_start,
            "end": Françoise_sea_start + len("Françoise"),
            "entity_type": "MISC",
            "score": 0.6,
            "source": "test",
        },
    ]

    anon_homophones, homophone_mapping = anonymizer.anonymize_with_tags(
        homophone_context, entities=homophone_entities
    )

    Françoise_entities = [
        info for info in homophone_mapping["entities"].values() if info["label"] == "PERSON"
    ]
    assert len(Françoise_entities) == 1, f"Les variantes de Françoise devraient fusionner: {Françoise_entities}"
    assert Françoise_entities[0]["canonical"] == "Françoise"
    assert {"Françoise", "Françoise"} <= set(Françoise_entities[0]["variants"])
    pseudo = Françoise_entities[0].get("pseudonym")
    assert pseudo, "Le pseudonyme doit être présent"
    assert anon_homophones.count(pseudo) == 2, "Le pseudonyme doit remplacer les deux mentions"
    assert "vie Françoise fragile" in anon_homophones, "Le contexte maritime ne doit pas être anonymisé"
    assert homophone_mapping["stats"]["total"] == 2, "Seules les mentions personnes doivent être comptées"

    print(f"✅ Détecté {len(mapping['entities'])} entités uniques")
    print(f"📊 Stats: {mapping['stats']}")
    print("✅ Test texte long OK")


def test_transformer_merges_similar_variants():
    """Les variantes proches doivent être fusionnées et corrigées."""

    text = "Françoise a parlé à Françoises. PENNYLANE rencontre Pény Lène."
    anonymizer = TransformerAnonymizer()

    Françoise_start = text.index("Françoise")
    Françoises_start = text.index("Françoises")
    penny_start = text.index("PENNYLANE")
    peny_start = text.index("Pény Lène")

    entities = [
        {"start": Françoise_start, "end": Françoise_start + len("Françoise"), "entity_type": "PERSON", "score": 0.95},
        {"start": Françoises_start, "end": Françoises_start + len("Françoises"), "entity_type": "PERSON", "score": 0.80},
        {"start": penny_start, "end": penny_start + len("PENNYLANE"), "entity_type": "PERSON", "score": 0.90},
        {"start": peny_start, "end": peny_start + len("Pény Lène"), "entity_type": "PERSON", "score": 0.90},
    ]

    anon_text, mapping = anonymizer.anonymize_with_tags(text, entities=entities)

    person_entities = [info for info in mapping["entities"].values() if info["label"] == "PERSON"]
    assert len(person_entities) == 2, f"Devrait fusionner les variantes, mapping={mapping}"

    canonicals = {info["canonical"] for info in person_entities}
    canonicals_norm = {unidecode(value).lower() for value in canonicals}
    assert "francoise" in canonicals_norm
    assert any("peny" in value or "penny" in value for value in canonicals_norm)
    assert unidecode(mapping["corrected_text"]).lower().count("francoise") == 2
    assert mapping["pseudonym_map"]
    assert "<" not in anon_text
    assert any(pseudo in anon_text for pseudo in mapping["pseudonym_map"].values())


def test_transformer_filters_pronouns_and_generic_words():
    """Les mots génériques ne doivent pas créer de faux positifs."""

    text = "Elle évoque une relance, Quand il faut, Moral n'est pas un prénom."
    anonymizer = TransformerAnonymizer()

    entities = []
    for word in ["Elle", "relance", "Quand", "Moral"]:
        start = text.index(word)
        entities.append(
            {"start": start, "end": start + len(word), "entity_type": "PERSON", "score": 0.9}
        )

    _, mapping = anonymizer.anonymize_with_tags(text, entities=entities)

    assert mapping["stats"]["total"] == 0
    assert mapping["entities"] == {}
    assert mapping["corrected_text"] == text


def test_deanonymization_exact():
    """Vérifie que la désanonymisation est exacte"""

    original = "Françoise et Micheline travaillent chez Top Avocats à Montpellier."

    anonymizer = TransformerAnonymizer(
        domain_entities={
            "PERSON": ["Françoise", "Micheline"],
            "ORGANIZATION": ["Top Avocats"]
        }
    )

    anon_text, mapping = anonymizer.anonymize_with_tags(original)
    restored = anonymizer.deanonymize(anon_text, mapping)

    # La désanonymisation doit être exacte
    assert restored == original, f"Désanonymisation incorrecte:\nOriginal: {original}\nRestored: {restored}"


def test_helper_function():
    """Test de la fonction helper pour compatibilité"""

    text = "Test avec Françoise et Top Avocats"
    domain = {"PERSON": ["Françoise"], "ORGANIZATION": ["Top Avocats"]}

    anon_text, mapping = run_transformer_anonymization(text, domain_entities=domain)

    assert "entities" in mapping
    assert "reverse_map" in mapping
    assert "pseudonym_reverse_map" in mapping
    assert anon_text != text


if __name__ == "__main__":
    # Run tests
    test_transformer_anonymization_basic()
    test_transformer_with_domain_entities()
    test_transformer_on_realistic_long_text()
    test_deanonymization_exact()
    test_helper_function()

    print("\n✅ Tous les tests passent!")
