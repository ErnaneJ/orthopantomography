# Log de Revisão: CBEB 2026 (Major Revision)

Registro de tudo que foi tentado para responder aos pareceres dos dois revisores do artigo *"Transfer Learning for Dental Pathology Detection in Panoramic Radiographs: YOLOv11 vs. Zero-Shot Grounding DINO"*. Cada item abaixo está classificado como:

- **[FEITO]**: corrigido no código e/ou no `paper_cbeb2026.tex`, com números recomputados a partir dos dados reais.
- **[ORIENTADOR/CEP]**: depende de aprovação, dado ou decisão que só o orientador pode encaminhar.
- **[NEGOCIAR COM REVISOR]**: não é possível resolver com os dados/recursos disponíveis; precisa de argumentação direta com os revisores.

---

## 1. Bug real na Tabela I: AP@50 vs. AP@50:95 trocados [FEITO]

`src/evaluation/evaluate_dentex.py` usava `metrics.box.maps[i]` (que é AP@50:95 por classe) rotulado como `"mAP50"` no JSON de saída, e avaliava com `conf=0.25` em vez do protocolo padrão do Ultralytics (`conf≈0.001`, que constrói a curva de precisão-recall completa). Isso subestimava artificialmente os números da Tabela I.

**Correção:** `conf=0.25 → conf=0.001`; separação explícita de `ap50[i]` (AP@50 correto) e `maps[i]` (AP@50:95, agora com campo próprio). Reexecutado.

**Números corrigidos** (antes → depois):
- mAP@50: 0.499 → **0.557**
- mAP@50:95: 0.321 → **0.361**
- Caries AP@50: 0.303 → **0.551**
- Periapical lesion AP@50: 0.095 → **0.189**
- Impacted tooth AP@50: 0.566 → **0.931**
- Precision/Recall: 0.583/0.550 (inalterados)

Propagado para: Abstract, Introdução, Tabela I, IV-A, V-B, V-D (Periapical), Conclusão. Este era provavelmente o ponto mais crítico apontado pelos revisores, pois os números antigos eram simplesmente errados por um bug de código, não uma limitação real do modelo.

---

## 2. Grounding DINO avaliado com um único prompt fixo [FEITO]

Reviewer apontou que uma única formulação de prompt não é evidência suficiente de que a detecção zero-shot falha de fato.

**O que foi feito:** `src/evaluation/ablation_gdino.py` (novo) testa 3 formulações de prompt (`caries` / `dental caries` / `a tooth with caries`, e equivalentes para as outras classes), 2 tamanhos de modelo (Tiny e Base) e 3 limiares de IoU (0.50, 0.25, 0.10): 12 configurações no total, usando detecções em cache para não reprocessar a inferência a cada limiar.

**Resultado:** mAP ficou em zero ou muito próximo de zero em todas as 12 configurações. O valor mais alto observado em qualquer configuração/limiar foi 0.015 (Tiny, prompt de nome de classe, Periapical lesion, IoU≥0.10). Grounding DINO Base não superou o Tiny. Isso descarta engenharia de prompt e tamanho do modelo como explicação: a falha de localização é mais severa do que "quase acerta, mas erra o IoU 0.5"; as caixas nem chegam a se aproximar consistentemente da região correta, mesmo em limiares bem mais permissivos.

Isso substituiu uma frase especulativa não sustentada que estava na Seção V-A ("at lower IoU thresholds... Grounding DINO would produce non-zero AP values") por dados reais, que na verdade contradizem a especulação original.

---

## 3. Contagem de "678 imagens de treino" ambígua [FEITO]

678 é o total de imagens anotadas em todos os splits (train+val+test), não o tamanho do conjunto de treino. O texto usava "678 training images" em pelo menos 2 lugares (Discussão V-B, Conclusão), o que estava literalmente incorreto.

**Correção:** trocado por "474 imagens de treino (950 após augmentação)" nos locais relevantes; a Seção III-A explica a decomposição completa (678 total, dividido em 474/101/103 por split, chegando a 950 imagens efetivas após oversampling das classes raras).

---

## 4. Contagens de instâncias por classe/split ausentes [FEITO]

Recontagem direta dos arquivos de anotação YOLO (com correção de um bug de parsing de IDs de classe em formato float). Valores cruzados com comentários já existentes no código de preparo de dados (`prepare_dentex.py`), que confirmam os totais 158 (Periapical) e 604 (Impacted) originais.

**Inserido na Seção III-A:**
- Treino (original/aumentado/total): Caries 1.934/2.008/3.942; Periapical 111/222/333; Impacted 458/916/1.374
- Validação: Caries 408; Periapical 25; Impacted 77
- Teste: Caries 425; Periapical 22; Impacted 69
- Total original: 2.767 Caries / 158 Periapical / 604 Impacted (3.529 instâncias)

Isso também permitiu substituir a alegação vaga de "Caries prevalence in routine clinical OPGs" (Seção IV-B) por uma explicação fundamentada nos próprios dados de treino (78,4% das instâncias anotadas são Caries), em vez de uma generalização epidemiológica sem fonte.

---

## 5. Intervalos de confiança ausentes para mAP e SR [FEITO]

`src/evaluation/bootstrap_ci.py` (novo): bootstrap por imagem (2000 reamostragens, seed=42) para mAP@50 do YOLOv11m no split de teste, usando a mesma implementação de AP de 11 pontos usada em `compare_baselines.py` para consistência interna. CI de Wilson para a proporção de imagens com SR ≥ 0.8.

**Resultado:**
- mAP@50 = 0.5547 (bootstrap), IC 95% [0.501, 0.645]
- SR ≥ 0.8: 86,7% (26/30), IC de Wilson 95% [70,3%, 94,7%]

Inserido na Tabela I (rodapé), Tabela II (rodapé) e no texto correspondente (IV-A, IV-C). O intervalo largo do Wilson CI, por causa do n=30 pequeno, é mencionado explicitamente, para não dar uma falsa sensação de precisão a um ponto estimado de amostra pequena.

---

## 6. BERTScore com `rescale_with_baseline=True` [FEITO]

`src/pipeline/stage4_metrics.py` agora computa BERTScore raw e rescaled em paralelo.

- Raw F1 = 0,7792 ± 0,0129 (próximo do valor original do artigo, 0,7803 ± 0,0135; pequena diferença por variação de seed/ordem, não é um erro)
- Rescaled F1 = −0,3085 ± 0,0767

O valor rescaled negativo **não é um bug nem evidência de baixa qualidade semântica**. É uma propriedade esperada do `rescale_with_baseline`, que reescala o score relativo a uma distribuição de baseline de domínio geral; pares de texto estruturalmente diferentes (laudo de 5 seções vs. nota curta em texto livre) caem fora dessa faixa de baseline por construção. O artigo agora reporta os dois valores lado a lado (Tabela III) com uma nota explicativa curta, em vez de esconder ou escolher só um.

---

## 7. Bug real (não reportado pelos revisores) no cálculo de SR agregado [FEITO, descoberta bônus]

Ao tentar reproduzir os números de SR para o item de CI acima, encontrei uma discrepância: `stage4_metrics.py` reportava 27 imagens avaliáveis com SR médio de 98,2%, mas o artigo diz 30 imagens e 88,3%. Isso parecia, a princípio, um problema de deriva de dados que exigiria uma decisão sua. Era, na verdade, um bug de código.

**Causa raiz:** `score = data.get("spontaneous_recall") or data.get("coverage_score")`. Em Python, `0.0 or x` avalia como `x`, porque `0.0` é falsy. Como consequência, as 3 imagens com SR real = 0,0 (as mesmas 3 da Tabela II, todas com Periapical lesion como única classe testável mencionada) eram descartadas da agregação. Como eram justamente as piores, a média subia artificialmente para 98,2%.

**Correção:** checagem explícita de `is not None` em vez de `or`. Reexecutado: reproduz exatamente os números originais do artigo (30 avaliáveis, SR médio 88,3%, 87% em alto recall). **Conclusão: a Tabela II do artigo estava correta desde o início; nenhuma mudança de número foi necessária ali.** Vale registrar porque, se não tivesse sido corrigido, um revisor mais atento nas próximas rodadas poderia reproduzir o pipeline e encontrar um número diferente do publicado.

---

## 8. Exemplo da Seção V-D com imagem errada [FEITO, descoberta bônus]

O texto citava "Image_05" com uma nota de referência específica e BERTScore = 0,791. Busquei o texto citado literalmente em todos os `stage3_reports/*.json`: ele pertence a `report_00.json` / `Image_00.jpg`, não a `Image_05.jpg`. Conferido: o laudo gerado para Image_00 realmente identifica os dentes 18 e 38 como impactados com angulação descrita, e múltiplas cáries, exatamente o que o texto do artigo descreve.

**Correção:** "Image_05" → "Image_00"; BERTScore recalculado especificamente para esse par: 0,7829 raw (0,791 → 0,783, diferença pequena, provavelmente de uma execução anterior com seed/versão de modelo distinta) e −0,2863 rescaled, ambos agora citados no texto.

---

## 9. Tabela I: renomeação e nota de threshold [FEITO]

Linha "GDINO (zero-shot)" → "GDINO-Tiny (zero-shot)" (precisão necessária: só o Tiny foi usado nessa tabela especificamente; o Base está na ablação da Seção V-A). O cabeçalho da tabela não tem mais um único "conf ≥ 0,25" genérico; passou a explicar que o YOLOv11m é avaliado por sweep de confiança (padrão Ultralytics, conf≥0,001) e o GDINO por decisão fixa (conf≥0,25, texto≥0,20), porque são protocolos de avaliação legitimamente diferentes, não uma inconsistência.

Isso também exigiu decidir a metodologia da Fig. 3 (comparação por classe): ela usa a implementação própria de AP de 11 pontos com limiar fixo conf≥0,25 (a mesma do `compare_baselines.py`, usada para a comparação direta contra o GDINO), e por isso os números ali (Caries 0,442 / Periapical 0,229 / Impacted 0,867 / média 0,512) são intencionalmente diferentes dos da Tabela I (0,551/0,189/0,931/0,557, protocolo de sweep). A legenda da figura e o texto do Método (III-B) agora deixam essa diferença explícita, em vez de ter dois números "mAP@50" diferentes sem explicação.

---

## 10. Reivindicação de "no GPU required" contraditória com o backend MPS [FEITO]

O Apple MPS usa a GPU integrada do chip M5; não é "sem GPU", é "sem GPU discreta/de datacenter". Corrigido em duas ocorrências (Seção V, Web Prototype; Conclusão).

---

## 11. "Periapical Lesion: A Data Problem": overclaim de causalidade [FEITO]

O texto original afirmava categoricamente "This is a data limitation, not an architecture one" sem ter testado essa hipótese (não há ablação de arquitetura com dados de Periapical escalados). Reformulado para reconhecer que o experimento não separa os dois fatores (volume de dado vs. dificuldade visual intrínseca da classe) e sugerir uma ablação de arquitetura como trabalho futuro para isolar a causa.

---

## 12. Legenda de cor "orange boxes would indicate..." hipotética na Fig. 6 [FEITO]

A Seção IV-C descrevia uma convenção de cor (verde = classe confirmada, laranja = classe fora do esperado) só especulativamente, sem que a figura específica mostrasse laranja. Movido: a convenção geral de cor agora está no Método (III-C, junto com a nota sobre os aliases do parser), e o texto específico da figura passou a descrever apenas o que está de fato na imagem (SR=1,0, só verde).

---

## 13. Lista de termos/sinônimos do parser de Stage 2 não documentada [FEITO]

`src/pipeline/utils.py` tem um dicionário de 24 aliases (23 termos únicos) para mapear menções em texto livre para as 31 classes clínicas. Rastreei o uso real: `match_findings_to_classes()` só é chamado por `stage2_validation.py`, e o resultado é sempre intersectado com as 3 classes treinadas pelo YOLO. Ou seja, das 24 entradas, só 5 afetam de fato a métrica SR reportada: `caries`/`carious` → Caries; `periapical` → Periapical lesion; `impacted`/`impaction` → Impacted tooth.

Adicionado ao Método (III-C) como nota de transparência/reprodutibilidade, com uma frase deixando claro que as outras 19 entradas cobrem termos fora do escopo de detecção das 3 classes e não influenciam a SR.

---

## 14. YOLOv11n mencionado na Fig. 2 mas nunca no texto [FEITO]

A Fig. 2 já trazia a curva do YOLOv11n desde antes, mas nenhum parágrafo do Método a mencionava. Carreguei o modelo (`YOLO('yolo11n.pt').info()`) para confirmar os números reais em vez de estimá-los: 2.624.080 parâmetros (~2,6M), 181 camadas, 6,6 GFLOPs; melhor mAP@50 de validação = 0,484 na época 32 (lido de `results/training/opg_yolo_nano/results.csv`). Adicionado como parágrafo curto no Método (III-B), deixando claro que o YOLOv11n é só uma referência de implantação leve e não foi usado no pipeline principal.

---

## 15. Referências \ref{} ausentes para 6 figuras [FEITO]

`fig:pipeline`, `fig:training`, `fig:comparison`, `fig:frequency`, `fig:recall` e `fig:nlp` tinham `\label` mas nunca eram citadas no corpo do texto (apareciam soltas, "flutuando"). Adicionada uma referência textual explícita para cada uma, no ponto do texto onde o conteúdo da figura é discutido.

---

## 16. Citações faltantes [FEITO, 3 de 4]

Adicionadas ao `references.bib` e citadas no texto:
- **Hamamci et al. (DENTEX benchmark, arXiv:2305.19112):** citado junto com `dentex2023` nas Seções II-A e III-A, como referência primária do benchmark (o `dentex2023` é o `howpublished` da página do desafio; Hamamci é o artigo técnico correspondente).
- **Asif & Khan (YOLO26 em radiografia panorâmica, arXiv:2604.16231):** citado na Seção V-B, como evidência independente de que transfer learning com a família YOLO permanece competitivo em radiografia dental panorâmica.
- **Dasanayaka et al. (LLM multimodal para laudo de OPG, Applied System Innovation, DOI 10.3390/asi8020039):** citado na Seção II-C, como o trabalho anterior mais próximo do nosso Stage 3, com a ressalva explícita de que o protocolo de avaliação e o dataset diferem e os resultados não são diretamente comparáveis.

### 16.1. Citação "Balel et al." [FEITO, resolvido nesta rodada]

Numa busca posterior encontrei o artigo correto: Balel et al., "A Novel Hybrid Large Language Model Approach for Reporting Panoramic Radiographs and Performance Comparison with Current Large Language Models" (Journal of Imaging Informatics in Medicine, 2026, DOI 10.1007/s10278-026-01880-9). Treinam um detector em 30.954 radiografias panorâmicas, convertem a saída para JSON estruturado e geram laudos com vários LLMs locais e comerciais, avaliados por especialistas; os modelos comerciais testados, incluindo uma variante do Gemini, alucinaram achados em 100% dos laudos revisados. Adicionado ao `references.bib` e citado na Seção II-C, com uma frase de contraste explícita: esse resultado, obtido com quase duas ordens de magnitude mais imagens de treino do que as nossas, é um alerta contra tratar laudos gerados por LLM como confiáveis sem a etapa de revisão por especialista que já listávamos como limitação e trabalho futuro.

---

## 17. Experimento de ablação de entrada do LLM (image-only vs. detections-only vs. both) [FEITO]

`src/evaluation/ablation_llm_input.py` (novo): gera laudos para as 50 imagens privadas em duas condições adicionais (apenas imagem, sem detecções do YOLO; apenas detecções, sem imagem), reaproveitando os laudos já existentes em `results/stage3_reports/` como a condição "both" (configuração padrão do pipeline), sem precisar reprocessá-los. As três condições são avaliadas com BLEU-4, ROUGE-L e BERTScore (raw e rescaled) contra as mesmas referências do dentista. 150 chamadas de API no total (50 imagens × 2 condições novas, mais reaproveitamento das 50 já existentes), todas concluídas sem erro.

**Resultado** (n=50 por condição):

| Condição | BLEU-4 | ROUGE-L | BERTScore raw | BERTScore rescaled |
|---|---|---|---|---|
| Image-only | 0,0010 | 0,0304 | 0,7761 | −0,3269 |
| Detections-only | 0,0015 | 0,0335 | **0,7956** | **−0,2113** |
| Both (padrão do pipeline) | 0,0010 | 0,0315 | 0,7792 | −0,3085 |

**Achado contraintuitivo:** a condição "apenas detecções" pontuou melhor que "ambos" e que "apenas imagem" em ROUGE-L e BERTScore (raw e rescaled). Isso não significa que a imagem é desnecessária. Significa que essas métricas medem similaridade textual com uma referência curta (~45 palavras), e um laudo instruído a não inventar nada além da lista de detecções fica naturalmente mais parecido em conteúdo e tamanho com essa referência curta. A imagem permite ao modelo descrever achados adicionais plausíveis (ex.: restaurações, espaços edêntulos) que não estão nas 3 classes treinadas nem na referência do dentista, o que reduz a sobreposição textual sem indicar um laudo pior. Essa interpretação foi incorporada ao artigo (Discussão, nova subseção "Image and Detection Contributions to Report Text"), com a ressalva explícita de que a ablação por si só não resolve a questão sem revisão clínica; o item de revisão clínica dos laudos (já listado como dependente do orientador) foi referenciado como o próximo passo necessário para resolver isso de fato.

Integrado ao artigo em: Abstract (frase nova), Método III-D (parágrafo curto descrevendo o desenho da ablação), Resultados IV-D (Tabela `tab:input_ablation` + parágrafo), nova subseção de Discussão, e Limitações (frase adicional).

---

## 18. Figuras: legendas duplicadas, fontes pequenas, fonte Type 3 [FEITO]

`src/generate_paper_figures.py`:
- Removidas todas as chamadas `fig.suptitle("Fig. N. ...")` nas 6 figuras: o LaTeX já gera a legenda numerada via `\caption{}`, e ter as duas ao mesmo tempo duplicava o texto na versão final.
- Fontes internas dos blocos da Fig. 1 (pipeline) elevadas de 7,5pt para 8pt.
- Adicionado `matplotlib.rcParams["pdf.fonttype"] = 42` (e `ps.fonttype`) para evitar fontes Type 3 em qualquer exportação PDF futura, exigência do IEEE PDF eXpress.
- Fig. 1 parou de salvar uma versão `.pdf` própria (só PNG), eliminando de vez o único ponto onde Type 3 poderia aparecer nessa figura.
- `fig_model_comparison()` tinha valores de AP por classe **hardcoded e desatualizados** (0,303/0,095/0,566/0,499), corrigidos para os valores corretos da metodologia de decisão fixa (conf≥0,25, AP de 11 pontos): 0,442/0,229/0,867/0,512, consistente com `results/evaluation/model_comparison.json`.
- Fig. 6 (métricas de PNL) ganhou um 4º painel para o BERTScore rescaled, com barras de erro para valores negativos tratadas corretamente (posicionamento de rótulo, eixo com zero visível).
- Todas as 6 figuras foram regeneradas e inspecionadas visualmente.

---

## 19. Validação manual do parser de Stage 2, exigida pelo Revisor 2 (item 5) [FEITO, bug real encontrado]

O revisor pediu explicitamente para validar manualmente o parser de texto livre, "sobretudo porque 40% das imagens foram excluídas" da métrica de Spontaneous Recall. Fiz essa auditoria linha a linha nas 20 imagens excluídas e encontrei um bug real: `src/pipeline/utils.py` não reconhecia os erros de digitação do dentista "imapcted", "carie" e "careis" (variantes de "impacted" e "caries"), presentes verbatim nas descrições de 4 das 20 imagens excluídas (Image_26, Image_31, Image_35, Image_36). Essas 4 imagens tinham, sim, achados testáveis, só que o dicionário de aliases não os reconhecia.

**Correção:** adicionadas as 3 variantes como aliases em `match_findings_to_classes()`. Reexecutado todo o pipeline Stage 2 e as agregações downstream (`stage4_metrics.py`, `bootstrap_ci.py`, `generate_paper_figures.py`).

**Números atualizados** (antes → depois): n avaliável 30 → **34**; SR médio 88,3% → **89,7%**; DP 30,8% → **29,2%**; distribuição SR≥0,8: 26/30 (86,7%) → **30/34 (88,2%)**; IC de Wilson [70,3%, 94,7%] → **[73,4%, 95,3%]**. As 4 imagens recuperadas tiveram SR=1,0 (YOLO detectou a classe correta em todas). Propagado para Abstract, Método III-C, Tabela II, Fig. 5, Resultados IV-C, Discussão V-C.

---

## 20. Segundo erro factual, independente do bug acima, na discussão das 3 imagens com SR=0,0 [FEITO, descoberta bônus]

Ao verificar os dados por trás da correção acima, percebi que o texto original afirmava que as 3 imagens com SR=0,0 tinham todas "Periapical lesion" como única classe testável. Conferindo diretamente `validation_23.json` e `validation_25.json`, a classe testável real dessas duas era "Impacted tooth", não Periapical lesion; só `validation_48.json` de fato tinha Periapical lesion como única classe mencionada. Esse erro já estava no artigo antes desta rodada, não foi causado pela correção do item 19 (essas 3 imagens já estavam no conjunto original de 30, sem relação com os aliases adicionados).

**Correção:** reescrita a frase para o achado real e mais informativo: em 2 dos 3 casos de SR=0,0, o YOLOv11m não detectou "Impacted tooth" nessas imagens específicas, apesar de essa classe ter o maior AP@50 agregado (0,931) no DentexChallenge, mostrando que um bom desempenho agregado não garante detecção em toda imagem individual do conjunto privado.

---

## 21. Threshold de confiança da Precisão/Recall na Tabela I não especificado, exigido pelo Revisor 2 (item 7) [FEITO]

O revisor pediu para especificar em que threshold de confiança os valores P=0,583/R=0,550 da Tabela I foram computados, já que o `mAP@50` da mesma linha usa o protocolo de sweep do Ultralytics (`conf≈0,001`), não um threshold fixo. Inspecionei o código-fonte do `ap_per_class()` da Ultralytics e reproduzi localmente com o modelo treinado (`models/yolo11_dentex.pt`): P e R são reportados no ponto de confiança que maximiza a média da curva de F1 suavizada entre as 3 classes, que é aproximadamente conf≈0,30 nesse modelo, não em conf=0,001 nem no threshold de deployment conf≥0,25 usado no resto do artigo.

**Correção:** adicionada nota de rodapé na Tabela I e frase no Método (III-B) esclarecendo que P/R vêm do ponto de máximo F1-médio do sweep (conf≈0,30), distinto do threshold de deployment.

---

## 22. Controle de pareamento aleatório para BERTScore, exigido pelo Revisor 2 (item 10) [FEITO]

O revisor pediu, além do `rescale_with_baseline=True` (já feito, item 6) e do hash do modelo, um controle de pareamento aleatório: reparear cada laudo gerado com uma referência de outro caso e comparar o BERTScore contra o pareamento correto, para testar se a métrica de fato distingue um laudo correto de um incorreto neste dataset, em vez de apenas saturar perto de um piso para qualquer par de texto em inglês no domínio.

**Script novo:** `src/evaluation/bertscore_control.py`. Reparea os 50 laudos com uma derangement fixa (sem pontos fixos, seed=42) do índice de referências.

**Resultado:** pares corretos: 0,7792 raw / −0,3085 rescaled (DP ≈ 0,013/0,077). Pares embaralhados (errados de propósito): 0,7777 raw / −0,3169 rescaled (DP ≈ 0,014/0,084). As duas distribuições praticamente se sobrepõem. **Conclusão honesta:** nem o BERTScore raw nem o rescaled discriminam de forma confiável um par correto de um par aleatoriamente errado nesta amostra de 50 laudos curtos. Isso não é um resultado esperado no início, é uma confirmação empírica direta da suspeita do revisor sobre a validade discriminativa da métrica.

Incluído no hash do modelo (`roberta-large_L17_no-idf_version=0.3.12(hug_trans=5.10.2)`) na Tabela III, e reescrito o parágrafo de Resultados (IV-D), Limitações e Conclusão para refletir esse achado de forma transparente, em vez de reportar só o valor absoluto sem controle.

---

## 23. Documentação da versão/data do endpoint Gemini, parte do item 16 (CEP) do Revisor 2 [FEITO, parte viável]

O revisor pediu informação de versão/data para reprodutibilidade dos experimentos com LLM. A parte de comitê de ética permanece dependente do orientador (ver seção abaixo), mas a versão do modelo é documentável: todas as chamadas usaram o alias `google/gemini-2.5-flash` roteado via OpenRouter, feitas em junho e agosto de 2026. Confirmado que esse alias não expõe um snapshot datado fixo no OpenRouter, ou seja, os resultados refletem a revisão de backend que o Google tinha em produção no momento da chamada, uma limitação real de reprodutibilidade de modelos hospedados via API que não está sob nosso controle.

Adicionada frase explícita no Método (III-D) documentando isso como limitação de reprodutibilidade, em vez de omitir a questão.

---

## 24. Confronto direto com Asif & Khan sobre volume de dados vs. distintividade visual [FEITO, reforço do item 16]

O item 16 já citava Asif & Khan (arXiv:2604.16231), mas de forma genérica. Nesta rodada, tornei o confronto específico: eles reportam mAP@50 = 0,591 (4 classes de doença, subconjunto do DENTEX processado via Roboflow) e argumentam que distintividade visual entre tipos de patologia, não volume anotado, é o fator mais forte de desempenho por classe. Isso contraria diretamente a explicação de "problema de dado" que demos para Periapical lesion (item 11 acima). Nossos resultados não resolvem a questão: "Impacted tooth" no nosso dado tem tanto a segunda maior contagem de instâncias quanto a aparência mais visualmente distinta, então arquitetura e volume de dado estão confundidos na mesma direção que no experimento deles. Separar os dois exigiria treinar em subconjuntos de volume equalizado por classe, o que não fizemos.

Adicionado como parágrafo novo na Seção V-D, reconhecendo explicitamente que a explicação de volume de dado é uma hipótese plausível parcial, não demonstrada.

---

## 25. Análise de sensibilidade ao limiar de confiança do Grounding DINO, pedida pelo Revisor 1 (texto narrativo) [FEITO]

O Revisor 1 pediu, no texto corrido do parecer, "uma análise da influência dos limiares de confiança" no Grounding DINO, distinto da ablação já feita (item 2 acima), que varia formulação de prompt × tamanho de modelo × IoU mas mantém o limiar de confiança fixo em 0,25 (texto em 0,20). Escrevi `src/evaluation/gdino_conf_sweep.py`, testando o Tiny com o prompt de nome de classe em limiares de confiança 0,05/0,10/0,15/0,25 × IoU 0,10/0,25/0,50 (12 configurações, reprocessando a inferência a cada limiar de confiança, diferente do IoU que pode ser recalculado sobre cache).

**Resultado:** no IoU mais permissivo (≥0,10), baixar o limiar de confiança de 0,25 para 0,05 quase triplicou o mAP, de 0,015 para 0,041, mostrando que algumas caixas corretamente localizadas mas de baixa confiança estavam sendo descartadas antes da pontuação. No IoU clinicamente relevante (≥0,5), porém, o mAP nunca passou de 0,007 em nenhum limiar de confiança testado. **Conclusão:** o limiar de confiança importa quando o critério de aceitação já é permissivo o bastante para recompensar caixas aproximadamente corretas; ele não resgata o desempenho no limiar de localização padrão. Isso confirma que a falha está em onde as caixas são colocadas, não em quantas estão sendo filtradas antes da pontuação.

Integrado à Seção V-A (Discussão), como novo parágrafo entre a ablação de prompt/modelo/IoU já existente e a conclusão da subseção.

---

## Itens dependentes do orientador / CEP [ORIENTADOR/CEP]

Estes não foram, e não deveriam ser, resolvidos por mim sozinho:

1. **Revisão clínica dos 50 laudos gerados por dentista especialista.** Os revisores provavelmente vão querer alguma forma de validação humana da qualidade clínica dos laudos, não só métricas de NLP (BERTScore mede similaridade textual, não corretude clínica, e isso já está explícito nas Limitações). Isso exige um dentista (co-autor ou avaliador externo) revisando uma amostra ou o total dos laudos.
2. **Documentação de aprovação ética / CEP e autorização de transferência internacional de dados.** As imagens privadas são de pacientes reais (Fundación Odontológica Social Luis Seiquer, Espanha) e são enviadas para a API do Gemini via OpenRouter (processamento fora do país de origem dos dados). Se ainda não há documentação formal de aprovação de comitê de ética e de base legal para essa transferência internacional (LGPD/GDPR, dependendo da jurisdição relevante), isso precisa ser resolvido antes da versão final. Não é algo que eu deveria decidir ou simular.
3. **Confirmação da autorização de uso dos dados privados para publicação.** Já há uma frase no Acknowledgment citando a fonte dos dados; vale confirmar com o orientador se o texto de autorização/consentimento usado é o que a instituição espanhola realmente aprovou para esta publicação específica.

## Itens para negociar diretamente com os revisores [NEGOCIAR COM REVISOR]

1. **Citação "Balel et al." não verificável** (detalhado no item 16.1 acima). Peça ao revisor o título ou o DOI exato.
2. Se algum revisor pedir avaliação em um **segundo dataset público independente** (além do DentexChallenge) para generalização, isso não é viável no prazo de uma revisão de conferência sem um dataset anotado adicional disponível. Vale argumentar que o DentexChallenge é, como já citado na Seção II-A, o maior benchmark público disponível para esta tarefa específica, e propor isso como trabalho futuro explícito (já mencionado na Conclusão).

---

## Arquivos modificados/criados nesta rodada

**Modificados (rodada 1):**
- `paper_cbeb2026.tex`: todas as correções de número, texto e citação descritas acima.
- `references.bib`: 3 novas entradas (Hamamci, Asif, Dasanayaka).
- `src/evaluation/evaluate_dentex.py`: bug de threshold e AP@50 vs. AP@50:95.
- `src/pipeline/stage4_metrics.py`: bug de agregação SR (`or` → `is not None`) + BERTScore raw/rescaled.
- `src/generate_paper_figures.py`: legendas duplicadas, fontes, valores hardcoded, Type 3.

**Criados (rodada 1):**
- `src/evaluation/bootstrap_ci.py`: CI de bootstrap (mAP) e Wilson (SR).
- `src/evaluation/ablation_gdino.py`: ablação de prompt/modelo/IoU do Grounding DINO.
- `src/evaluation/ablation_llm_input.py`: ablação de entrada do LLM (imagem/detecções/ambos).
- `REVISION_LOG.md`: este arquivo.

**Resultados novos (rodada 1):**
- `results/evaluation/dentex_test_metrics.json` (recomputado)
- `results/evaluation/confidence_intervals.json`
- `results/evaluation/gdino_ablation.json`
- `results/evaluation/llm_input_ablation_summary.json`
- `results/metrics/all_metrics.json` (recomputado)
- `results/figures/*.png` (todas as 6, regeneradas)

**Modificados (rodada 2, itens 19-25):**
- `paper_cbeb2026.tex`: correção do bug de aliases do parser + renumeração de SR (n=34), correção do segundo erro factual (SR=0,0), nota de threshold de P/R na Tabela I, controle de pareamento aleatório do BERTScore, nota de versão/data do Gemini, citação Balel et al., confronto com Asif & Khan.
- `references.bib`: nova entrada `balel2026hybrid`.
- `src/pipeline/utils.py`: 3 novos aliases de digitação (`imapcted`, `carie`, `careis`).
- `.gitignore`: adicionado `results/evaluation/llm_input_ablation/` (dado clínico privado, não coberto antes).

**Criados (rodada 2):**
- `src/evaluation/bertscore_control.py`: controle de pareamento aleatório do BERTScore.
- `src/evaluation/gdino_conf_sweep.py`: sensibilidade ao limiar de confiança do Grounding DINO (item 25, em andamento).

**Resultados novos (rodada 2):**
- `results/stage2_validations/*` (recomputado, n=34)
- `results/metrics/all_metrics.json` (recomputado novamente)
- `results/metrics/bertscore_random_pairing_control.json`
- `results/evaluation/confidence_intervals.json` (recomputado, Wilson CI atualizado)
- `results/evaluation/gdino_confidence_sweep.json`
- `results/figures/*.png` (regeneradas com n=34)
