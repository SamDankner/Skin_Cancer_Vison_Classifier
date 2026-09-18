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
or the authors' listed institution, respecting their terms.

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
doi:10.1002/ski2.120, states that the data are available from the corresponding
author on reasonable request. The study reports smartphone clinical images,
healthy and abnormal labels, and patient information. Request through the
publication's corresponding-author route.

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

## Fitzpatrick17k

Fitzpatrick17k annotations are available from the official project repository,
but images retain their original-source terms and the annotation file does not
reliably declare clinical modality. Supply an official clinical-photo-only bundle
and provenance before enabling ingestion; never use its dermoscopic images.
