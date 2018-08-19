---
title: "Road to AWS Certification"
layout: post
date: 2018-08-13 22:10
tag: AWS
headerImage: true
projects: false
hidden: false # don't count this post in blog pagination
description: "AWS Journey"
category: blog
---

 
Nowadays, Kaggle have been releasing such large datasets that made it difficult to processing on my 16 GB RAM machine. So I have always 
wanted to explore and use cloud computing but I never found the time during the semester. I recently moved back to Hong Kong after finishing 
my Master's degree, I saw AWS was hosting their first summit in Hong Kong. Luckly, I was able to sign up for GameDay and the Summit. 

However, I have never used AWS before so I spent a day or two reading up their services and tutorials before GameDay :sweat_smile:  

![jpg](lawko698.github.io/assets/images/2018-08-13-photos/IMAG1055)

During GameDay, there were a set of challenges teams needed to complete. 

1. Set up a static website using S3, then switch to a dynamic website using an EC2 instance. Perhaps the simplest task.
2. Use Amazon Rekognition to train a model to recognize our imaginary CEO's face. Set up a Lambda trigger on the S3 bucket when a photo 
of the CEO gets uploaded. Probably one of the hardest task since there was limited documentation to fill in the python code. 
3. Big Data task. Use EMR or other services to query the large dataset.
4. CI/CD Task to detect and automate uploads from github as it changes every 5 mins. We also need to test the build befor deployment.
This part uses AWS code commit. 
5. I can't remember exactly the topic but it uses AWS code pipeline for the task.

We managed to complete the first three tasks, but we struggled to complete 4 and 5 as it heavily relies on codecommit and code pipeline which
none of us were familiar with. In the end, we obtained 5th place which I felt was pretty good given we were all new to AWS. The only thing
I would complain about is the sticker they gave us. It was simply too large to stick on my laptop :laughing: .

![jpg](lawko698.github.io/assets/images/2018-08-13-photos/IMAG1056)

Then came the Summit. The most interesting part was the invited speakers that explained their AWS architecture. 
We had [Thomson Reuters](https://www.thomsonreuters.com/en.html), [TVB](http://tvb.com/) and [Asia Miles](https://www.asiamiles.com/).
I thought Asia Mile's speaker was one of the best presentations out of all of the speakers. For those who don't know, Asia Miles 
incoporated blockchain technology into their [new mobile application](http://fintechnews.hk/5140/blockchain/cathay-pacific-asia-miles-blockchain/).
They utilized AWS to build their infrastructure without needing to buy servers or plan for future capacity. The presentation was really well done as he
also included AWS's motto of being a [Day One company](https://blog.aboutamazon.com/company-news/2016-letter-to-shareholders).

After the summit, I decided to study for the AWS Solution Architect- Associate Certificate. I spent around 2 weeks of study and I managed to complete the exam.
One of the largest problems with studying for the certification exam is how fast technology changes. I wasn't sure how many of the new service updates 
they have incoporated into the exam, but from what I have seen so far, the questions were relatively free of out-dated material. It just comes to show how old technology
becomes redundant really quickly in the world of IT.
