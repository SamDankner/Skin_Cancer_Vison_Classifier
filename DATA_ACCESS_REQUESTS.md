# Data access requests

This project accepts only ordinary clinical or macro photographs. Never upload
dermoscopy, microscopy, pathology slides, or DDI data to development folders.
Raw images are ignored by Git and must remain subject to their source terms.

## ImageQX / Imagine teledermatology collection

The study is R. Jalaboi et al., *Explainable Image Quality Assessments in
Teledermatological Photography*, Telemedicine and e-Health (2023),
doi:10.1089/tmj.2022.0405. It describes 36,509 mobile photographs, including
dermatologist-defined lesion, healthy-skin, poor-quality, and no-skin classes.
It would provide valuable within-source healthy-versus-lesion comparisons and
reported age, sex, and body-part metadata. No official image download endpoint
was located during this audit; request access through the corresponding author
Raluca Jalaboi at `rjal@dtu.dk` or the authors' listed institution, respecting
their terms. The article is public, but that does not itself make the underlying
private user photographs downloadable.

Draft request:

> I am conducting non-commercial research on robust clinical-photo skin-image
> classification. May I request access to the ImageQX/Imagine image files and
> accompanying labels (healthy skin, lesion, poor quality, no skin), patient or
> encounter identifiers suitable for grouped splitting, and permitted age, sex,
> and body-site metadata? Please also share the applicable license, permitted
> uses, redistribution restrictions, and any required ethics or data-use terms.

## Muhaba et al. smartphone dataset

K. A. Muhaba et al., *Automatic skin disease diagnosis using deep learning from
clinical image and patient information*, Skin Health and Disease (2022),
doi:10.1002/ski2.81, states that the data are available from the corresponding
author on reasonable request. The study reports smartphone clinical images,
healthy and abnormal labels, and patient information. Request through Gizeaddis
Lamesgin Simegn at `gizeaddis.lamesgin@ju.edu.et` (the publication also lists
`1time.et@gmail.com`).

Draft request:

> I am conducting non-commercial research on leakage-safe classification of
> ordinary clinical photographs. Could you share the approved image files,
> healthy/abnormal labels, patient IDs suitable for grouped splitting, and any
> permitted age, sex/gender, anatomical-site, symptom, and clinical metadata?
> Please include the data-use license, allowed uses, and redistribution terms.

## ENCoDE / PhysioNet

ENCoDE, *mEasuring skiN Color to correct pulse Oximetry DisparitiEs*, PhysioNet
v1.0.0 (2024), doi:10.13026/mcgk-1s42, provides smartphone images from released
non-biometric body locations plus demographics and skin-tone measurements. It
requires PhysioNet credentialing, required training, acceptance of the data-use
agreement, and download through the official PhysioNet project page. It is not
a dermatologist-labelled healthy-skin dataset. If approved and locally supplied,
ingest it only as `normal_label_strength=auxiliary` / nonlesional appearance
data unless an approved label source justifies more.

The protocol excluded wounds/bruising and several causes of discoloration, but
it was designed for skin-tone and pulse-oximetry research rather than a
dermatologist no-visible-lesion label. Consequently ENCoDE remains auxiliary and
is not eligible for the core gate by default.

## Additional-source search outcome

The focused search did not identify another immediately downloadable,
authoritatively labelled, same-source smartphone collection with both focal
lesions and true no-visible-lesion controls. Collections of inflammatory disease
images, random portraits/body photographs, stock imagery, synthetic skin,
dermoscopy, and pathology are out of scope. New focal-lesion-only clinical sets
can improve positive diversity but do not solve the gate's negative-class or
source-shortcut problem; they should be evaluated separately before ingestion.

## Public hard-negative sources

MCSI is acquired only from the official Zenodo v2 record (`10.5281/zenodo.8360076`). Its 400 labelled close-skin images are split evenly among normal, acne, chickenpox, and mpox. The release reports diverse public sources and cropping/zooming, so the pipeline preserves its documented labels but assigns `moderate`, rather than `strong`, reliability.

MSLD v2.0 is publicly hosted on Kaggle under CC BY-NC 4.0. Acquisition uses only the user's ordinary Kaggle CLI credentials; the importer accepts the published Original Images and excludes every MATLAB-augmented folder. Filenames encode disease, patient, and image number. If credentials are unavailable, preparation reports `KAGGLE_AUTHENTICATION_REQUIRED` rather than bypassing login.

The Mendeley `MonkeyPox` aggregate (`10.17632/st6kggjr23.1`) is not admitted automatically. Its record says it was curated and extended from existing sources and it provides no defensible patient grouping, so it remains under provenance and duplicate review.

## Fitzpatrick17k

Fitzpatrick17k annotations are available from the official project repository,
but images retain their original-source terms and the annotation file does not
reliably declare clinical modality. Supply an official clinical-photo-only bundle
and provenance before enabling ingestion; never use its dermoscopic images.
