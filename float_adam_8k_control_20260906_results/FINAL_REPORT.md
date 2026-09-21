[RESULT]

Early stopping at epoch 23; best epoch 13. Patience 10, maximum 60. The original process continued uninterrupted past epoch 8; no second training run or checkpoint restart occurred.

| Metric | Best epoch 13 | Last epoch 23 |
|---|---:|---:|
| Train loss | 3.5846 | 2.7286 |
| Train all-token accuracy (%) | 29.5114 | 41.3693 |
| Train content accuracy (%) | 22.4871 | 35.4832 |
| Held-out all-token accuracy (%) | 20.7072 | 20.0370 |
| Held-out content accuracy (%) | 13.2417 | 12.5286 |
| Held-out content loss | 5.2562 | 5.7095 |
| EOS recall (%) | 94.0000 | 93.7500 |
| EOS false-positive rate (%) | 1.7061 | 1.5533 |
| Position 0 accuracy (%) | 12.0000 | 12.5000 |

Train metrics above cover all 7,600 rows. Per-epoch train monitoring in history.json covers the fixed first 400 training rows. Every held-out evaluation covers all 400 validation rows.

[HELD-OUT SOURCE DEPENDENCE]

| Content accuracy (%) | Best epoch 13 | Last epoch 23 |
|---|---:|---:|
| Real | 13.2417 | 12.5286 |
| Shuffled | 7.6649 | 6.6208 |
| Zeroed memory | 6.1370 | 5.3476 |
| Real - shuffled (pp) | 5.5768 | 5.9078 |
| Real - zeroed (pp) | 7.1047 | 7.1811 |

| Best-checkpoint condition | Position 0 accuracy (%) | All-token loss | Content loss |
|---|---:|---:|---:|
| Real | 12.0000 | 4.8047 | 5.2562 |
| Shuffled | 2.0000 | 5.4620 | 5.9728 |
| Zeroed memory | 3.5000 | 6.5609 | 6.5473 |

Exact JA-KO pair overlap: 0. Three Japanese source strings occur in both splits with different Korean targets. Excluding these three validation rows leaves 397 examples: real content accuracy 13.2037%, shuffled 7.5998%, zeroed 6.0389%; gaps 5.6039 / 7.1648 pp; chrF 6.6499. This supplemental evaluation did not select the checkpoint.

[GENERATION]

| Metric | Best epoch 13 | Last epoch 23 |
|---|---:|---:|
| chrF | 6.6607 | 6.5579 |
| Character BLEU, add-one smoothing | 6.2736 | 6.2725 |
| Mean generated characters | 14.9300 | 14.4100 |
| Mean reference characters | 18.3900 | 18.3900 |
| Mean generated content tokens | 8.4650 | 8.3300 |
| Korean characters (%) | 71.5506 | 73.2998 |
| Replacement characters (%) | 0.0000 | 0.0000 |
| Distinct outputs / 400 | 388.0000 | 399.0000 |
| Unigram token repetition (%) | 19.8464 | 17.2569 |
| Bigram token repetition (%) | 14.1661 | 11.7326 |
| Generated/reference character-length ratio | 0.8119 | 0.7836 |

Best-checkpoint shuffled-source greedy chrF: 4.1895; character BLEU: 3.4097. chrF and character BLEU use the existing repository implementations; BLEU is not word-tokenized SacreBLEU.

First 10 held-out rows, in saved split order; outputs from the selected checkpoint:

| Japanese source | Korean reference | Greedy output |
|---|---|---|
| 怖くて泣いていたら | 그러면 겁에 질리고 울었겠지요. | 그러니까 출에 싫어보세요. |
| これを知ること無しに何を予想していたでしょうか? | 이 사실을 모른다면 어떤 예측을 했을까요? | 이걸 어떻게 이걸 어떻게 할까요? |
| どうやってやるのか? | 어떻게 이런 일을 할 수 있을까요? | 어떻게 해야할까요? |
| 倫理的英雄を賞賛してください | 인정해 주세요. | 그래서 저는 떨도 놓합니다. |
| これこそがタイムマシンなのです | 이것이 우리의 타임머신입니다. | 이 사진은 이미입니다. |
| 私は単純で明快な説明を信じません | 저는 뻔한 이야기는 믿지 않습니다. | 그 사람은 별로 능을 담기 때문입니다. |
| これは単なる飾り言葉ではありません | 그건 그저 싯구가 아니에요. | 정말 멋진 전혀 밖이 아닙니다. |
| さぁ 立って | 자, 일어나. | 탄이 놓치고 |
| ほとんどの人は みな合理的なのです | 거의 모든 이들이 합리적입니다. | 그리고 이 사람들을 쉬운 일입니다. |
| 「女性 全員ですか?」 | "노년 여성들 전부요? | "그래?" |

[VERDICT]

CASE 3: float training itself failed.

Under the declared stopping rule, final training content accuracy was 35.4832%. This run did not establish held-out translation success or an integer-specific cause.

Verification: 8 focused tests passed; protected Python files and the tokenizer retain their audited hashes; the selected checkpoint reproduces its recorded held-out metrics exactly. Acceptance leakage remains 0/30.

Artifacts: audit.json, split.json, smoke_state.pt, full_started.json, history.json, best.pt, last.pt, result.json, shared_source_exclusion.json. best.pt and last.pt contain model weights only; the complete on-disk optimizer/RNG checkpoint remains the epoch-1 smoke state. No later trajectory resume is claimed.

Runner: ../float_adam_8k_control_20260906.py; tests: ../test_float_adam_8k_control_20260906.py.
