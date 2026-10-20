# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Stay involved in the conversation and address any review feedback.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Community-contributed models
The `contrib/` folder is home to models contributed by the community. They are maintained by their contributors and may
have less testing or performance optimization than the models built into the plugin under `vllm_neuron/model/`.
Contributing a model to `contrib/` is intentionally lightweight for now, and we plan to make the process more seamless
over time. Where your change fits depends on what you're adding:

- **A new model architecture not already supported in `vllm_neuron/model/`** — contribute it under
  [`contrib/`](contrib/README.md). Before you start, check the `contrib/` folder and existing PRs for an implementation
  of the same architecture to avoid duplicated effort. See [contrib/README.md](contrib/README.md) for the folder
  structure, required tests, and model-specific guidelines.
- **An improvement to a model already in `contrib/`** — if you can collaborate with the original contributor, we
  encourage working together on their implementation and submitting a new PR against the existing `contrib/` folder. If
  you are not able to collaborate with the original PR author for that model and you are adding specific performance
  optimizations or features, you can contribute a separate implementation rather than changing the existing one in
  place. Be specific about what yours adds in the PR description so we can see how it differs from the existing one.
- **Framework changes needed to support your model architecture** — these can go in the same PR as your model. Note that
  we may scope the PR based on those changes (for example, ask to split or defer some of them) so we can review and land
  it in the right increments.
- **A feature you want to contribute to a model that requires significant framework-level changes and is broadly
  applicable to other model architectures** — we encourage you to file a GitHub feature request before you contribute,
  so we can discuss the design together. We may not be able to accept large framework changes as-is, but the feature
  request helps us find the right path.
- **A feature or fix for a model already maintained in `vllm_neuron/model/`** — we recommend opening a GitHub issue
  first. These models have testing criteria that must pass before changes are applied, so aligning early helps make sure
  your effort isn't wasted.

### Maintaining a contributed model
Contrib models are community-maintained. If you contribute a model, we encourage you to help maintain it over time —
testing it against newer versions of the `vllm-neuron` plugin and opening PRs to keep it working as the plugin evolves.
This keeps your model usable for others.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.