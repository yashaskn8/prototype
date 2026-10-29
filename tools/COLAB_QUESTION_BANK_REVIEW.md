# Servy — Colab question-only review

This tool only checks and filters a **locally uploaded, synthetic question-only JSON**. It does not connect to Servy's Django database, access private manuals, upload results, train models or calculate technical accuracy.

## Use

Obtain \`servy_colab_1200_questions.json\` separately from the person who generated it. The JSON must contain a top-level \`questions\` list with \`id\`, \`asset_id\`, \`product\`, \`module\`, \`question\`, \`question_sha256\`, and \`coverage_status\` for each record.

Run (locally or in Google Colab after cloning this **public** GitHub repository and uploading only the question-only JSON):

\`\`\`bash
python tools/colab_question_bank_review.py --input servy_colab_1200_questions.json
python tools/colab_question_bank_review.py --input servy_colab_1200_questions.json --asset-id 60 --show 5
python tools/colab_question_bank_review.py --input servy_colab_1200_questions.json --asset-id 60 --module troubleshooting --export vrf_troubleshooting_questions.json
\`\`\`

Each question's full SHA-256 is verified to detect the previous 200-character question-truncation error.

## Scope and limitations

The proposed bank covers 12 registered assets with 100 candidate questions per asset, split into installation/commissioning (30), maintenance (30), and troubleshooting (40). These are **hypothetical adversarial question prompts** based only on equipment names, not gold answers produced from company manuals. Based on the last local inventory, only assets 59, 60, and 61 had knowledge-base documents; most asset questions must therefore be handled as unsupported or pending documentation.

**Do not use the 1,200 prompts as 1,200 verified answerable examples.** Validate ground truth against approved documentation. Obtain authorization for each tested customer/asset; the earlier selected scope covered asset 60 only. The test-labeled asset without a customer must not contribute to an equipment accuracy metric. Keep question banks used for training separate from held-out evaluation cases.

**GitHub repository is public:** do not commit source-derived answer keys, proprietary documents, customer details, actual RAG responses, credentials or unredacted evaluation reports. Do not upload company data to personal Google Colab without explicit cloud-processing permission.

The utility does not modify Servy's retrieval model or RAG behavior. Make and validate any future production changes against the latest local uncommitted source files, not a stale GitHub snapshot.
