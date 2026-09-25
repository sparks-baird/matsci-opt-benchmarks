# Edison literature query: LLM-style multi-fidelity hyperparameter transfer reference

- Task id: `3a490c3c-5c23-4448-80a0-ca4c85e3787c`
- Job: `job-futurehouse-paperqa3`
- Status: `success`
- Created: 2026-09-25T20:52:51.766596Z
- Has successful answer: True
- Total cost: None; total queries: None

Edison output below is unedited except that en and em dashes were converted to plain punctuation. The unmodified task record is in `llm-fidelity-reference.json`. Verification note: the NeurIPS 2021 proceedings title of the recommended paper is "Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (first author listed as Ge Yang); the "Tensor Programs V:" prefix belongs to the extended arXiv version (arXiv:2203.03466, author listed as Greg Yang). NeurIPS does not assign DOIs; the arXiv DOI is 10.48550/arXiv.2203.03466. See `Yang2021_muP` and `Klein2017_FABOLAS` in `../new_refs_staging.bib`.

## Query

I need the single best peer-reviewed (or widely cited) reference that describes tuning hyperparameters on cheaper, smaller proxy training runs (for example, smaller models with fewer parameters, or subsets of the training data) and then transferring or extrapolating those hyperparameters to the expensive full-scale training run, ideally in large language model (LLM) or large neural network training. The analogy we want to support in a paper is: find optimal hyperparameters on a 100M-parameter model to decide which hyperparameters to use on a 1B-parameter model. In our paper this is framed as multi-fidelity hyperparameter optimization, where the training-set fraction is the fidelity parameter. Please: (1) name the single best reference for this practice (tune on small proxy, transfer or extrapolate to large-scale training), give the full citation with DOI and/or arXiv ID, and explain in two sentences why it fits the analogy; (2) name one runner-up reference that specifically uses training-set size (a subset fraction of the training data) as the fidelity parameter in Bayesian or multi-fidelity hyperparameter optimization (for example, something like FABOLAS), with the full citation and DOI and/or arXiv ID. Prefer primary sources over blog posts, and state the publication venue for each.

## Formatted answer (with references)

Question: I need the single best peer-reviewed (or widely cited) reference that describes tuning hyperparameters on cheaper, smaller proxy training runs (for example, smaller models with fewer parameters, or subsets of the training data) and then transferring or extrapolating those hyperparameters to the expensive full-scale training run, ideally in large language model (LLM) or large neural network training. The analogy we want to support in a paper is: find optimal hyperparameters on a 100M-parameter model to decide which hyperparameters to use on a 1B-parameter model. In our paper this is framed as multi-fidelity hyperparameter optimization, where the training-set fraction is the fidelity parameter. Please: (1) name the single best reference for this practice (tune on small proxy, transfer or extrapolate to large-scale training), give the full citation with DOI and/or arXiv ID, and explain in two sentences why it fits the analogy; (2) name one runner-up reference that specifically uses training-set size (a subset fraction of the training data) as the fidelity parameter in Bayesian or multi-fidelity hyperparameter optimization (for example, something like FABOLAS), with the full citation and DOI and/or arXiv ID. Prefer primary sources over blog posts, and state the publication venue for each.

## 1. Best reference: small-model tuning transferred to large-model training

**Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., & Gao, J. (2021). “Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.” *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*, 17084-17097. arXiv:2203.03466.** No conventional proceedings DOI appears to have been assigned. (lingle2404alargescaleexploration pages 9-11, li2506maximalupdateparametrization pages 11-12)

This is the closest direct support for the proposed analogy: its **µTransfer** procedure tunes hyperparameters on a cheap, smaller model under the maximal-update parametrization and transfers them zero-shot to the full-sized network, without tuning the latter directly. The paper validates this on Transformers, including transfer from **13M to 350M parameters for BERT** and from **40M parameters to a 6.7B-parameter GPT-3 model**, making it stronger and more specific support for a 100M→1B analogy than generic multi-fidelity-HPO papers. (yang2021tuninglargeneural pages 1-2, yang2021tuninglargeneurala pages 1-2)

A qualification worth preserving in the citing text is that µTransfer does not claim that every conceivable hyperparameter transfers unchanged: it principally supports optimization, initialization, and parameter-multiplier hyperparameters under the µP scaling rules, while regularization parameters such as dropout and weight decay may require separate treatment. (yang2021tuninglargeneural pages 1-2)

## 2. Runner-up: training-set fraction as fidelity

**Klein, A., Falkner, S., Bartels, S., Hennig, P., & Hutter, F. (2017). “Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets.” In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*, Proceedings of Machine Learning Research, **54**, 528-536. arXiv:1605.07079; arXiv DOI: https://doi.org/10.48550/arXiv.1605.07079.** (klein2017fastbayesianoptimization pages 7-9, klein2017fastbayesianoptimization pages 1-2)

This paper introduces **FABOLAS**, the canonical match to your paper’s particular multi-fidelity framing. It treats training-set size (the size or fraction of a randomly sampled data subset) as a continuous environmental/fidelity variable, jointly models validation loss and evaluation cost across subset sizes, and chooses inexpensive subset evaluations for their information about the best hyperparameters on the **full dataset**. (klein2017fastbayesianoptimization pages 7-9, klein2017fastbayesianoptimization pages 1-2)

| Rank/use | Reference | Peer-reviewed venue | Proxy/fidelity mechanism | Identifiers |
|---|---|---|---|---|
| **Best match: model-scale transfer** | Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* | NeurIPS 2021; *Advances in Neural Information Processing Systems* 34:17084-17097 | Tune a smaller µP proxy and zero-shot transfer hyperparameters to the large model; demonstrated at 13M→350M parameters for BERT and 40M→6.7B for GPT-3 (yang2021tuninglargeneural pages 1-2, lingle2404alargescaleexploration pages 9-11) | arXiv:2203.03466 (li2506maximalupdateparametrization pages 11-12) |
| **Runner-up: data-size fidelity** | Klein et al., *Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets* (FABOLAS) | AISTATS 2017; PMLR 54:528-536 | Treat training-set subset size/fraction as a continuous fidelity/environmental variable; model loss and cost at cheap subset sizes and extrapolate to the full dataset (klein2017fastbayesianoptimization pages 1-2) | arXiv:1605.07079; DOI: 10.48550/arXiv.1605.07079 (klein2017fastbayesianoptimization pages 7-9) |


*Table: Comparison of the strongest model-size transfer reference with the canonical Bayesian-optimization reference that uses training-set size as fidelity.*

**Recommendation:** cite **Yang et al.** for the general claim that hyperparameters can be tuned on a much smaller neural-network proxy and transferred to a billion-scale target, and cite **Klein et al.** alongside it when describing your specific implementation of *training-set fraction as the fidelity parameter*.

References

1. (lingle2404alargescaleexploration pages 9-11): Lucas D. Lingle. A large-scale exploration of μ-transfer. ArXiv, Jan 2404. URL: https://doi.org/10.48550/arxiv.2404.05728, doi:10.48550/arxiv.2404.05728. This article has 1 citations.

2. (li2506maximalupdateparametrization pages 11-12): Shanda Li, Shinjae Yoo, and Yiming Yang. Maximal update parametrization and zero-shot hyperparameter transfer for fourier neural operators. ArXiv, Jun 2506. URL: https://doi.org/10.48550/arxiv.2506.19396, doi:10.48550/arxiv.2506.19396. This article has 4 citations.

3. (yang2021tuninglargeneural pages 1-2): G Yang, E Hu, I Babuschkin, S Sidor, X Liu, D Farhi, N Ryder, J Pachocki, W Chen, and J Gao. Tuning large neural networks via zero-shot hyperparameter transfer. Unknown journal, 2021.

4. (yang2021tuninglargeneurala pages 1-2): G Yang, E Hu, I Babuschkin, and S Sidor. Tuning large neural networks via zero-shot hyperparameter transfer. Unknown journal, 2021.

5. (klein2017fastbayesianoptimization pages 7-9): Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. Preprint, Jan 2017. URL: https://doi.org/10.48550/arxiv.1605.07079, doi:10.48550/arxiv.1605.07079. This article has 953 citations.

6. (klein2017fastbayesianoptimization pages 1-2): Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. Preprint, Jan 2017. URL: https://doi.org/10.48550/arxiv.1605.07079, doi:10.48550/arxiv.1605.07079. This article has 953 citations.

## Evidence contexts

### yang2021tuninglargeneural pages 1-2 (score 9)

Citation: G Yang, E Hu, I Babuschkin, S Sidor, X Liu, D Farhi, N Ryder, J Pachocki, W Chen, and J Gao. Tuning large neural networks via zero-shot hyperparameter transfer. Unknown journal, 2021.
DOI: None

The excerpt directly identifies Yang et al.’s paper as “Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer,” authored by Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. It places the work at the 35th Conference on Neural Information Processing Systems (NeurIPS 2021), but provides no DOI or arXiv identifier and does not explicitly state peer-review status beyond the proceedings citation. The method is µTransfer: the target network is parameterized with Maximal Update Parametrization, a smaller model is tuned, and the resulting hyperparameters are copied to the larger target without directly tuning it. Examples include transferring from 13M parameters to BERT-large (350M) and from 40M parameters to GPT-3’s 6.7B model. The table indicates that optimization-related hyperparameters, initialization, and parameter multipliers are transferable; empirical transfers also include width, depth, batch size, training time, and sequence length. Regularization such as dropout and weight decay is listed as not transferable. The supplied excerpt contains no information about Klein et al.’s FABOLAS paper, its authorship metadata, DOI/arXiv ID, venue, peer-review status, or its use of training-set fraction as a fidelity variable.

### ma2026μpscalingsmallmodels pages 67-69 (score 3)

Citation: Yuxin Ma, Nan Chen, M. Díaz, Soufiane Hayou, Dmitriy Kunisky, and Soledad Villar. Μpscaling small models: principled warm starts and hyperparameter transfer. ArXiv, 2026. URL: https://doi.org/10.48550/arxiv.2602.10545, doi:10.48550/arxiv.2602.10545. This article has 4 citations.
DOI: 10.48550/arxiv.2602.10545

The excerpt provides only limited, indirect evidence concerning Yang et al. (2021). It states that the authors’ experiments validate hyperparameter transfer across model widths and explicitly relates this behavior to Yang et al. (2021), described as work demonstrating similar behavior without upscaling. The reported experiments vary learning-rate and injected-noise hyperparameters across MLP and GPT-2 widths, with optimal hyperparameters generally transferring across widths. However, this excerpt does not provide Yang et al.’s exact author list, title, publication venue, year beyond the in-text 2021 reference, DOI, or arXiv identifier. It also does not establish the requested details about smaller proxy models, full-size Transformers, parameter sizes, or the complete set of transferred hyperparameters in Yang et al.’s work. Klein et al. and FABOLAS are not mentioned, so there is no evidence here concerning training-set size or subset fraction as a fidelity/environmental variable, extrapolation to the full dataset, Bayesian optimization, bibliographic metadata, peer review, or proceedings venue. No information in the excerpt identifies whether either cited work was peer-reviewed or appeared in conference proceedings.

### lingle2404alargescaleexploration pages 7-9 (score 4)

Citation: Lucas D. Lingle. A large-scale exploration of μ-transfer. ArXiv, Jan 2404. URL: https://doi.org/10.48550/arxiv.2404.05728, doi:10.48550/arxiv.2404.05728. This article has 1 citations.
DOI: 10.48550/arxiv.2404.05728

The excerpt provides only partial evidence relevant to Yang et al.’s work. It identifies Yang et al. as the first report of hyperparameter transfer via µP and states that their largest target model had 6.7B parameters, with results also involving 13B-parameter baselines. However, it does not give Yang et al.’s exact author list, title, publication venue, year, DOI, or arXiv identifier. It also does not explicitly state the complete proxy-to-target procedure or enumerate all transferred hyperparameters; instead, it contrasts Yang et al.’s broader hyperparameter-transfer work with the present article’s narrower evaluation of learning-rate transfer. The excerpt contains no information about Klein et al.’s FABOLAS paper: there is no evidence concerning its authorship, title, venue, year, DOI/arXiv ID, dataset-size or subset-fraction fidelity variable, extrapolation to the full dataset, Bayesian optimization, peer-review status, or proceedings venue. Consequently, bibliographic and peer-review/proceedings claims for both requested papers cannot be confirmed from this excerpt, although the Yang-related scale and µP-transfer context are partially supported.

### kim2608letsscalestep pages 8-10 (score 2)

Citation: Nayeon Kim, Hojin Lee, Yunju Bak, Jaesun Park, and Boseop Kim. Let's scale step by step: compute-efficient hyperparameter transfer for large-scale mixture-of-experts. ArXiv, Aug 2608. URL: https://doi.org/10.48550/arxiv.2608.20061, doi:10.48550/arxiv.2608.20061. This article has 0 citations.
DOI: 10.48550/arxiv.2608.20061

The excerpt provides only limited, indirect evidence relevant to Yang et al. It cites Yang and Hu (2021) and Yang et al. (2022) as work enabling zero-shot transfer of optimal hyperparameters, including learning rate, across model width through parameterization changes and scaling of learning rates or logits for specific tensors. However, it does not provide the exact title “Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer,” complete author list, publication venue, year, DOI or arXiv identifier, specific proxy and full-model parameter sizes, or a complete list of transferred hyperparameters. The excerpt contains no information about Klein et al.’s “Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets” or FABOLAS: it does not mention training-set size as a fidelity/environmental variable, subset fractions, extrapolation to the full dataset, Bayesian optimization, bibliographic metadata, peer review, or proceedings venue. It likewise does not establish whether either cited work was peer-reviewed or identify its proceedings venue. The surrounding paper is a COLM 2026 conference paper, but that venue applies to the excerpted source, not necessarily to Yang et al. or Klein et al.

### yang2021tuninglargeneurala pages 1-2 (score 8)

Citation: G Yang, E Hu, I Babuschkin, and S Sidor. Tuning large neural networks via zero-shot hyperparameter transfer. Unknown journal, 2021.
DOI: None

The excerpt provides strong evidence for Yang et al.’s paper title, its full author list (including Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao), and publication as part of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021). It does not provide a DOI or a specific arXiv identifier; it only refers generally to arxiv.org. The method is µTransfer: the target network is parametrized in Maximal Update Parametrization (µP), hyperparameters are tuned on a smaller width/depth proxy, and the tuned values are copied to the full-size model without directly tuning it. Examples include transfer from a 13M-parameter model to BERT-base/BERT-large and from 40M parameters to the 6.7B GPT-3 model. Listed transferable categories include optimization-related hyperparameters, initialization, parameter multipliers, width, depth, batch size, training time, and sequence length, with caveats; regularization such as dropout and weight decay is listed as non-transferable. The excerpt contains no information about Klein et al., FABOLAS, dataset-size fidelity, extrapolation, Bayesian optimization, DOI/arXiv ID, peer-review status, or proceedings venue.

### klein2017fastbayesianoptimization pages 7-9 (score 8)

Citation: Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. Preprint, Jan 2017. URL: https://doi.org/10.48550/arxiv.1605.07079, doi:10.48550/arxiv.1605.07079. This article has 953 citations.
DOI: 10.48550/arxiv.1605.07079

The excerpt provides direct bibliographic evidence for the Klein et al. work: the authors are Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter; the title is “Fast bayesian optimization of machine learning hyperparameters on large datasets”; it is identified as a January 2017 preprint; and the supplied DOI/arXiv URL is https://doi.org/10.48550/arxiv.1605.07079 (arXiv:1605.07079). It does not identify a peer-reviewed proceedings venue, so the excerpt supports describing this version as a preprint rather than assigning a conference publication venue. For FABOLAS, dataset size is explicitly treated as an additional continuous input variable, functioning as the fidelity or environmental variable. The method evaluates configurations on subsets of the data, models evaluation time, and selects evaluations for information gained per unit time about the best hyperparameters on the full dataset. The conclusion explicitly characterizes FABOLAS as Bayesian optimization based on entropy search and reports experiments with support vector machines and deep neural networks. The supplied excerpt contains no information about Yang et al., proxy-model parameter sizes, transferred hyperparameters, or that paper’s venue, year, DOI/arXiv identifier, peer-review status, or proceedings publication.

### ghosh2512understandingthemechanisms pages 1-2 (score 3)

Citation: Nikhil Ghosh, Denny Wu, and Alberto Bietti. Understanding the mechanisms of fast hyperparameter transfer. ArXiv, Dec 2026. URL: https://doi.org/10.48550/arxiv.2512.22768, doi:10.48550/arxiv.2512.22768. This article has 8 citations.
DOI: 10.48550/arxiv.2512.22768

The excerpt provides limited, indirect evidence relevant to Yang et al.’s work. It cites Yang et al. (2022) as empirical support for fast hyperparameter transfer under the Maximal Update Parameterization (µP). It explains that this approach enables practitioners to tune hyperparameters on smaller proxy models and apply the selected values to larger-scale training runs with minimal performance loss. The discussion specifically frames model-width scaling and describes learning rate as a scale-independent hyperparameter multiplied by a width-dependent factor. It also states that, under µP, optimal hyperparameters are expected to become asymptotically scale-independent, allowing tuning on a fixed grid across scales. The excerpt does not provide the exact Yang et al. title, complete author list, publication venue, publication year beyond the citation’s 2022 label, DOI or arXiv identifier, concrete parameter sizes, or a complete list of transferred hyperparameters. It contains no information about Klein et al., FABOLAS, training-set size or subset fraction as a fidelity variable, extrapolation to the full dataset, Bayesian optimization, peer-review status, or proceedings venue.

### zhou2601howtoset pages 1-2 (score 3)

Citation: Yunhua Zhou, Shuhao Xing, Junhao Huang, Xipeng Qiu, and Qipeng Guo. How to set the learning rate for large-scale pre-training? ArXiv, Jan 2026. URL: https://doi.org/10.48550/arxiv.2601.05049, doi:10.48550/arxiv.2601.05049. This article has 5 citations.
DOI: 10.48550/arxiv.2601.05049

The excerpt provides only indirect, contextual evidence about Yang et al.’s 𝜇Transfer work. It states that 𝜇Transfer searches for hyperparameters, including learning rate, on a smaller proxy model and transfers them to a target model. It also says that the original formulation was limited to extrapolating model width. These passages support the general proxy-to-target transfer concept requested for Yang, but they do not provide Yang et al.’s exact author list, paper title, publication venue, publication year, DOI, or arXiv identifier. They also do not specify the parameter sizes of the proxy and target models, identify a full-size Transformer explicitly, or enumerate all hyperparameters shown to transfer. The excerpt contains no substantive information about Klein et al. or FABOLAS: it does not mention training-set size or subset fraction as a fidelity/environmental variable, extrapolation to the full dataset, Bayesian optimization, bibliographic metadata, peer-review status, or proceedings venue. Consequently, the supplied text cannot confirm the requested details for either paper beyond the broad proxy-model transfer principle associated with 𝜇Transfer.

### klein2017fastbayesianoptimization pages 1-2 (score 8)

Citation: Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. Preprint, Jan 2017. URL: https://doi.org/10.48550/arxiv.1605.07079, doi:10.48550/arxiv.1605.07079. This article has 953 citations.
DOI: 10.48550/arxiv.1605.07079

The excerpt provides evidence for Klein et al.’s work: Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter authored “Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets.” It identifies the publication as appearing in the Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), in 2017. The supplied bibliographic information gives DOI 10.48550/arxiv.1605.07079 and arXiv identifier arXiv:1605.07079. The method is FABOLAS, a Bayesian optimization procedure designed for expensive large-dataset training. It explicitly uses dataset size as an additional input or fidelity/environmental variable: the optimizer chooses the size of a randomly subsampled training set for each evaluation. FABOLAS models loss and training time as functions of dataset size, explores configurations on small, cheaper subsets, and extrapolates performance to the full dataset. The stated objective remains performance at Nsub = N, the full dataset, rather than optimization of the subset size itself. The excerpt describes the AISTATS proceedings venue but does not explicitly state peer-review status. It contains no information about Yang et al.’s authorship, title, venue, year, DOI/arXiv ID, proxy-model parameter sizes, or transferable hyperparameters.

### yang2021tuninglargeneurala pages 13-14 (score 2)

Citation: G Yang, E Hu, I Babuschkin, and S Sidor. Tuning large neural networks via zero-shot hyperparameter transfer. Unknown journal, 2021.
DOI: None

The excerpt is a references section from the 2021 manuscript and does not provide a title page, proceedings citation, DOI information, or publication metadata for “Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.” It lists earlier Tensor Programs papers by Greg Yang (Tensor Programs I, II, and III) as arXiv works, along with related work on feature learning and architectural universality, but it does not list Tensor Programs V itself or identify a NeurIPS 2022 venue, conference number, or arXiv identifier such as 2203.03466. The excerpt also contains no entry for FABOLAS and therefore cannot verify its publication in Proceedings of AISTATS 2017, PMLR volume 54, pages 528-536, nor establish whether it has a conventional DOI in addition to an arXiv DOI. The only potentially related hyperparameter-tuning reference shown is Yogatama and Mann’s “Efficient Transfer Learning Method for Automatic Hyperparameter Tuning,” published in AISTATS with pages 1077-1085 and a PMLR URL; this is a different work and should not be conflated with FABOLAS.

### li2506maximalupdateparametrization pages 11-12 (score 6)

Citation: Shanda Li, Shinjae Yoo, and Yiming Yang. Maximal update parametrization and zero-shot hyperparameter transfer for fourier neural operators. ArXiv, Jun 2506. URL: https://doi.org/10.48550/arxiv.2506.19396, doi:10.48550/arxiv.2506.19396. This article has 4 citations.
DOI: 10.48550/arxiv.2506.19396

The excerpt’s references identify the work as “Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer,” authored by Greg Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao. It records the publication type as an arXiv preprint and gives the identifier arXiv:2203.03466, dated 2022. The excerpt does not provide a NeurIPS proceedings citation, confirm that the paper appeared at NeurIPS 2022 or the 36th conference, or state a DOI. It also contains no reference or metadata for FABOLAS, so it cannot verify the claimed AISTATS 2017 Proceedings of Machine Learning Research volume 54, pages 528-536 details or determine whether a conventional DOI exists beyond an arXiv DOI. The surrounding bibliography includes other conference references and arXiv entries, but none supplies the missing publication metadata for either requested verification.

### shen2024powerschedulera pages 11-12 (score 6)

Citation: Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya Prasad, Adriana Meza Soria, David D. Cox, and Rameswar Panda. Power scheduler: a batch size and token number agnostic learning rate scheduler. ArXiv, Aug 2024. URL: https://doi.org/10.48550/arxiv.2408.13359, doi:10.48550/arxiv.2408.13359. This article has 27 citations.
DOI: 10.48550/arxiv.2408.13359

The excerpt provides reference-list metadata for the work titled “Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.” It identifies the authors as Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao, and cites it as an arXiv preprint from 2022 with identifier arXiv:2203.03466. However, this excerpt does not verify publication at NeurIPS 2022, does not mention the 36th Conference on Neural Information Processing Systems, and supplies no proceedings citation or DOI information. It also contains no reference to FABOLAS, so it cannot verify that FABOLAS appeared in Proceedings of AISTATS 2017, PMLR volume 54, pages 528-536, nor determine whether a conventional DOI exists beyond an arXiv DOI. Further bibliographic sources, such as the official NeurIPS proceedings, PMLR record, or publisher metadata, are required for those confirmations.

### lingle2404alargescaleexploration pages 9-11 (score 8)

Citation: Lucas D. Lingle. A large-scale exploration of μ-transfer. ArXiv, Jan 2404. URL: https://doi.org/10.48550/arxiv.2404.05728, doi:10.48550/arxiv.2404.05728. This article has 1 citations.
DOI: 10.48550/arxiv.2404.05728

The excerpt provides a proceedings reference for Greg Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao, and coauthors. It identifies the work as “Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer” and cites it in Advances in Neural Information Processing Systems, volume 34, pages 17084-17097, published by Curran Associates in 2021. Therefore, this excerpt does not support the claim that the paper was published at NeurIPS 2022 or that it appeared in the 36th conference; instead, it explicitly gives volume 34 and 2021. The cited entry includes a NeurIPS proceedings PDF URL but does not provide an arXiv identifier, including 2203.03466, and does not list a DOI. The excerpt contains no title-page metadata or independent arXiv record for this paper. It also contains no reference to FABOLAS, AISTATS 2017, PMLR volume 54, pages 528-536, or any DOI information for FABOLAS. Consequently, the requested FABOLAS publication details and whether it has a conventional DOI beyond an arXiv DOI cannot be verified from this excerpt.

### ma2026μpscalingsmallmodels pages 17-20 (score 7)

Citation: Yuxin Ma, Nan Chen, M. Díaz, Soufiane Hayou, Dmitriy Kunisky, and Soledad Villar. Μpscaling small models: principled warm starts and hyperparameter transfer. ArXiv, 2026. URL: https://doi.org/10.48550/arxiv.2602.10545, doi:10.48550/arxiv.2602.10545. This article has 4 citations.
DOI: 10.48550/arxiv.2602.10545

The excerpt provides a references-list citation for “Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer” by Greg Yang and coauthors. It identifies the work as appearing in the Proceedings of the 35th International Conference on Neural Information Processing Systems (NeurIPS), pages 17084-17097, in 2021. Therefore, this excerpt does not support the claim that it was published at NeurIPS 2022, which was the 36th conference. The supplied text does not include a title page, an arXiv identifier, or a DOI for this paper, so it cannot verify the believed arXiv number 2203.03466 or determine whether a DOI exists. FABOLAS is not mentioned anywhere in the excerpt, and no metadata is provided for an AISTATS 2017 publication, PMLR volume 54, pages 528-536, or any conventional DOI beyond an arXiv DOI. Verification of those details requires consulting the paper’s official title page, the PMLR proceedings record, or the relevant arXiv/Crossref entries.

### xiao2409rethinkingconventionalwisdom pages 23-25 (score 5)

Citation: Lechao Xiao. Rethinking conventional wisdom in machine learning: from generalization to scaling. ArXiv, Sep 2409. URL: https://doi.org/10.48550/arxiv.2409.15156, doi:10.48550/arxiv.2409.15156. This article has 31 citations.
DOI: 10.48550/arxiv.2409.15156

The excerpt’s references identify the work as “Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer,” authored by Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. It cites the work as an arXiv preprint from 2022 with identifier arXiv:2203.03466. This excerpt does not provide a NeurIPS 2022 proceedings citation, confirmation that it appeared at the 36th NeurIPS conference, a title-page record, or a DOI. Therefore, those details cannot be verified from the supplied pages. FABOLAS is not mentioned anywhere in the excerpt, so its alleged AISTATS 2017 publication details (Proceedings of AISTATS, volume 54, pages 528-536) or the existence or nonexistence of a conventional DOI beyond an arXiv DOI also cannot be verified here. The supplied material is a bibliography section rather than the definitive publication records or title pages requested.
