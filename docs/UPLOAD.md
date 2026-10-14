# Uploading the supporting release

Commit the contents of `IL-property-prediction/` as the repository root. The prepared repository ZIP is an alternative transfer container; unpack it before making the initial commit. Do not commit the surrounding `supporting_release/` folder, the larger archives, or your full research workspace.

The repository contains compressed saved predictions and seven small final encoders. The largest file is below 25 MiB. The three larger files should be attached to a GitHub release or deposited in a persistent data archive, then linked from the repository. GitHub blocks ordinary Git files above 100 MiB and recommends keeping repositories small; see [GitHub's large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

After creating an empty repository using your chosen account and repository name, from the prepared folder:

```bash
git init -b main
git add .
git commit -m "Add manuscript supporting data and code"
git remote add origin https://github.com/JiahuiChen1998/IL_Contrastive_Multitask.git
git push -u origin main
```

The target repository was created by the author at https://github.com/JiahuiChen1998/IL_Contrastive_Multitask. These commands are for a new local checkout; clone and preserve any existing remote history when the remote already has commits.

Create a versioned release such as `v1.0.0` and attach:

- `graph_checkpoints_v1.zip`
- `pretrain_train_set.txt.gz`
- `pretrain_val_set.txt.gz`
- `large_assets.json` and `SHA256SUMS.txt` from the surrounding delivery folder

Preserve the corpus attribution to the ILBERT authors in the release description and archive metadata. The code MIT license does not assign a separate new license to the supplied corpus. Retain the upstream dataset/model notices.

The actual repository URL is recorded in `CITATION.cff` (`repository-code`) and the README. If you deposit the versioned release in an archive with a DOI, add that DOI to `CITATION.cff` and replace the manuscript's data-availability placeholder with the public URLs. A DOI or publication status has not been invented in this package.

Suggested data-availability wording after those URLs exist:

> The original property workbooks are available from the ILTransR repository. Curated observations and fixed complete-IL split manifests, model code and configurations, numerical source tables, and saved validation/test predictions are available in the supporting repository [repository URL]. The versioned graph checkpoints and exact pretraining input files supplied by the ILBERT authors are available at [release or archive URL/DOI].
