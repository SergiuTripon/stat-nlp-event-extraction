package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 05/11/2015.
 */

object Features {

  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Trigger Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
    val token = thisSentence.tokens(begin) //first token of Trigger
    feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0 //word feature
    feats.toMap
  }
  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Argument Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0
    feats.toMap
  }

  def myNaiveBayesTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val prefix = "trigger::"
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey, Double]

    val token = thisSentence.tokens(begin) //first token of Trigger

    //features provided
    feats += FeatureKey(prefix + "label bias", List(y)) -> 1.0 //bias feature
    feats += FeatureKey(prefix + "first trigger word", List(token.word, y)) -> 1.0 //word feature

    //features developed
    val tokenizerStem = token.stem.split("-")
    for(segment <- tokenizerStem) {
      if (y.toLowerCase.startsWith(segment)) {
        feats += FeatureKey(prefix + "trigger dictionary starts with", List(token.word, segment, y)) -> 1.0 //segment word stem as part of trigger dictionary feature
      }
    }

    val mentions = thisSentence.mentions
    feats += FeatureKey(prefix + "first mention length", List(mentions.length.toString, y)) -> 1.0 //first mention length feature

    feats.toMap

  }

  def myPerceptronTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val prefix = "trigger::"
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]

    val token = thisSentence.tokens(begin) //first token of Trigger

    //features provided
    feats += FeatureKey(prefix + "label bias", List(y)) -> 1.0 //bias feature
    feats += FeatureKey(prefix + "first trigger word", List(token.word, y)) -> 1.0 //word feature

    //features developed
    feats += FeatureKey(prefix + "first trigger pos", List(token.pos, y)) -> 1.0 //pos feature
    feats += FeatureKey(prefix + "first trigger stem", List(token.stem, y)) -> 1.0 //stem feature
    feats += FeatureKey(prefix + "first trigger length", List(token.word.length.toString, y)) -> 1.0 //first trigger length feature

    //tokens
    if ((begin > 0) && (begin < thisSentence.tokens.length)) {

      val leftToken = thisSentence.tokens(begin-1)
      val rightToken = thisSentence.tokens(begin+1)

      if (leftToken.pos == "NN" && token.pos.startsWith("N") && rightToken.pos == "IN") {
        feats += FeatureKey(prefix + "left right pos on nouns", List(leftToken.pos, token.pos, rightToken.pos, y)) -> 1.0 //left right pos on nouns feature
      }

      if (leftToken.pos == "JJ" && token.pos.startsWith("V") && rightToken.pos == "NN") {
        feats += FeatureKey(prefix + "left right pos on verbs", List(leftToken.pos, token.pos, rightToken.pos, y)) -> 1.0 //left right pos on verbs feature
      }

      if (!Set("NN").contains(leftToken.pos) && token.pos == "IN" && Set("DT","NN", "JJ").contains(rightToken.pos)) {
        feats += FeatureKey(prefix + "none trigger event", List(token.word, y)) -> 1.0 //none trigger event feature
      }

      if (token.pos.equals("VBD") && Set("TO").contains(rightToken.pos)) {
        feats += FeatureKey(prefix + "none trigger event to", List(token.word, y)) -> 1.0 //none trigger event to feature
      }

      //feats += FeatureKey("token unigram on the left", List(leftToken.word, y)) -> 1.0 //unigram on the left feature
      feats += FeatureKey(prefix + "token bigram on the left", List(leftToken.word, token.word, y)) -> 1.0 //bigram on the left feature
      //feats += FeatureKey("token unigram on the right", List(rightToken.word, y)) -> 1.0 //unigram on the right feature
      feats += FeatureKey(prefix + "token bigram on the right", List(token.word, rightToken.word, y)) -> 1.0 //bigram on the right feature

      if (token.word.startsWith("-") || token.word.contains("-")) {
        feats += FeatureKey(prefix + "token startswith/contains", List(token.word, rightToken.word, y)) -> 1.0 //token startswith/contains feature
      }

    }

    val tokenizerStem = token.stem.split("-")
    for(segment <- tokenizerStem) {
      if (y.toLowerCase.startsWith(segment)) {
        feats += FeatureKey(prefix + "trigger dictionary starts with", List(token.word, segment, y)) -> 1.0 //segment word stem as part of trigger dictionary feature
      }
    }

    val tokenizer = token.word.split("-")
    for(segment <- tokenizer) {
      val index_ = if(y.indexOf('_') > 0) y.indexOf('_') else 0
      val labelGold = y.substring(index_ + 1)
      if (labelGold.toLowerCase.endsWith(segment)) {
        feats += FeatureKey(prefix + "trigger dictionary ends with", List(token.word, segment, y)) -> 1.0 //segment word stem as part of trigger dictionary feature
      }
    }

    //mentions
    val mentions = thisSentence.mentions
    val leftMention = mentions.filter(m => m.begin <= begin)
    val rightMention = mentions.filter(m => m.begin >= end)

    feats += FeatureKey(prefix + "first mention length", List(mentions.length.toString, y)) -> 1.0 //first mention length feature

    if (leftMention.nonEmpty) {
      val distanceLeft = (begin - leftMention.last.begin).toString
      feats += FeatureKey(prefix + "distance to left mention", List(distanceLeft, y)) -> 1.0 //distance to left mention feature
    }

    if (rightMention.nonEmpty) {
      val distanceRight = (rightMention.head.begin - end).toString
      feats += FeatureKey(prefix + "distance to right mention", List(distanceRight, y)) -> 1.0 //distance to right mention feature
    }

    //deps
    val deps = thisSentence.deps
    val depHead = deps.filter(dh => {dh.head == begin})
    val depMod = deps.filter(dm => {dm.mod == begin})

    depHead.foreach(dh => {
      feats += FeatureKey(prefix + "dep head", List(dh.label, y)) -> 1.0 //dep head feature
    })

    depMod.foreach(dm => {
      feats += FeatureKey(prefix + "dep mod", List(dm.label, y)) -> 1.0 //dep mod feature
    })

    feats.toMap
  }

  def myNaiveBayesArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val prefix = "arg::"
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    val feats = new mutable.HashMap[FeatureKey,Double]
    val token = thisSentence.tokens(begin) //first word of argument

    //features provided
    feats += FeatureKey(prefix + "label bias", List(y)) -> 1.0 //label bias feature
    feats += FeatureKey(prefix + "is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0 //is protein_first trigger word feature

    //features developed
    if (!x.isProtein) {
      feats += FeatureKey(prefix + "first argument pos", List(token.pos, y)) -> 1.0 //first argument pos feature
    }

    feats += FeatureKey(prefix + "first token of event pos", List(eventHeadToken.pos, y)) -> 1.0 //first token of event pos feature

    val mentions = thisSentence.mentions
    feats += FeatureKey(prefix + "first mention length", List(mentions.length.toString, y)) -> 1.0 //first mention length feature

    val distance = (begin - event.begin).toString
    feats += FeatureKey(prefix + "distance between begin and event.begin", List(distance, y)) -> 1.0 //distance between begin and event.begin feature

    val deps = thisSentence.deps
    val depHead = deps.filter(dh => {dh.head == begin})
    val depMod = deps.filter(dm => {dm.mod == begin})

    depHead.foreach(dh => {
      val mentions = thisSentence.mentions.filter(_.begin == dh.head)
      feats += FeatureKey(prefix + "dep head to protein", List(dh.label, mentions.length.toString, y)) -> 1.0 //dep head to protein feature
    })

    depMod.foreach(dm => {
      val mentions = thisSentence.mentions.filter(_.begin == dm.mod)
      feats += FeatureKey(prefix + "dep mod to protein", List(dm.label, mentions.length.toString, y)) -> 1.0 //dep mod to protein feature
    })

    feats.toMap

  }

  def myPerceptronArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val prefix = "arg::"
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    val feats = new mutable.HashMap[FeatureKey,Double]
    val token = thisSentence.tokens(begin) //first word of argument

    //features provided
    feats += FeatureKey(prefix + "label bias", List(y)) -> 1.0 //label bias feature
    feats += FeatureKey(prefix + "first argument word", List(token.word, y)) -> 1.0 //first argument word feature
    feats += FeatureKey(prefix + "is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0 //is protein_first trigger word feature

    //features developed
    feats += FeatureKey(prefix + "first token of event pos", List(eventHeadToken.pos, y)) -> 1.0 //first token of event pos feature
    feats += FeatureKey(prefix + "is protein_first", List(x.isProtein.toString, y)) -> 1.0 //is protein_first feature

    if (begin > 0) {
      val leftEvent = thisSentence.tokens(begin-1)
      feats += FeatureKey(prefix + "unigram on the left using begin", List(leftEvent.word, y)) -> 1.0 //unigram on the left using begin feature
      feats += FeatureKey(prefix + "bigram on the left using begin", List(leftEvent.word, eventHeadToken.word, y)) -> 1.0 //bigram on the left using begin feature
    }

    if (begin < thisSentence.tokens.length) {
      val rightEvent = thisSentence.tokens(begin+1)
      feats += FeatureKey(prefix + "unigram on the right using begin", List(rightEvent.word, y)) -> 1.0 //unigram on the right using begin feature
      feats += FeatureKey(prefix + "bigram on the right using begin", List(eventHeadToken.word, rightEvent.word, y)) -> 1.0 //bigram on the right using begin feature
    }

    if (event.begin > 0) {
      val leftEvent = thisSentence.tokens(event.begin-1)
      feats += FeatureKey(prefix + "unigram on the left using event.begin", List(leftEvent.word, y)) -> 1.0 //unigram on the left using event.begin feature
      feats += FeatureKey(prefix + "bigram on the left using event.begin", List(leftEvent.word, eventHeadToken.word, y)) -> 1.0 //bigram on the left using event.begin feature
    }

    if (event.begin < thisSentence.tokens.length) {
      val rightEvent = thisSentence.tokens(event.begin+1)
      feats += FeatureKey(prefix + "unigram on the right using event.begin", List(rightEvent.word, y)) -> 1.0 //unigram on the right using event.begin feature
      feats += FeatureKey(prefix + "bigram on the right using event.begin", List(eventHeadToken.word, rightEvent.word, y)) -> 1.0 //bigram on the right using event.begin feature
    }

    val distance = (begin - event.begin).toString
    feats += FeatureKey(prefix + "distance between begin and event.begin", List(distance, y)) -> 1.0 //distance between begin and event.begin feature

    //deps
    val deps = thisSentence.deps
    val depHead = deps.filter(dh => {dh.head == begin})
    val depMod = deps.filter(dm => {dm.mod == begin})

    if(depHead.isEmpty || depMod.isEmpty) {
      feats += FeatureKey(prefix + "None coming/going arguments due to No dependencies", List("ZeroDeps", y)) -> 1.0 //No related dependencies with Candidate
    }

    depHead.foreach(dh => {
      feats += FeatureKey(prefix + "dep head", List(dh.label, y)) -> 1.0 //dep head feature
    })

    depMod.foreach(dm => {
      feats += FeatureKey(prefix + "dep mod", List(dm.label, y)) -> 1.0 //dep mod feature
    })

    feats.toMap

  }


}
