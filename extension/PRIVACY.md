# Lacuna privacy policy

*Last updated: 28 October 2026*

Lacuna is a Chrome extension that compares a job posting with your CV, on your device.

## What Lacuna handles

- **Your CV.** The text you paste or load in Lacuna's settings. It is stored in your browser's extension storage (`chrome.storage.local`) and never leaves your device.
- **The job page you score.** When you click "Score this page" (or, if you turn on automatic scoring, when you open a page on a supported job site), Lacuna reads the posting's text from that tab. The text is processed in your browser and is not sent anywhere.
- **Your scored jobs.** The title, company, location, link, fit score and gaps of each job you score are kept in extension storage so the "Ranked" list works. You can clear the list at any time.
- **Settings.** Your model choice and preferences, in extension storage.

## What Lacuna sends over the network

Only one thing: on first use, Lacuna **downloads the model files** (the Lacuna Celer or Lacuna Altus model, its tokenizer and configuration) from Hugging Face (`huggingface.co`). These are ordinary file downloads; no CV, page content or personal data is included in them. The files are then kept in your browser's cache so later scoring works offline.

Lacuna has no servers, no accounts, no analytics, no tracking and no ads. Nothing you do in Lacuna is collected, sold or shared, and none of it is used to train models.

## Removing your data

Delete your CV in Settings, clear the Ranked list, use "Delete downloaded models", or remove the extension: Chrome then deletes everything Lacuna stored.

## Contact

Questions: open an issue at https://github.com/cucoleadan/lacuna-ai/issues
