---
layout: post
title: Pytorch images models
abstract: An overview of one of the best library in Pytorch. It help us to understand and quickly deploy SOTA models in a few line code.
tags: [flow, focus, work, productivity]
category: book
mathjax: true
words: 1149-8760
time: 5-33
thumbnail: /images/deep_work.jpg
---

<div class="noprint" markdown="1">

# {{ page.title }}

Pytorch image models is good start if you want to build something with currently good performance. You can have enough control but the accuracy is still good.
Many models pretrained in ImageNet that is one of best metric in image classification or port from Tensorflow.

## Part 1: Why should i use this?

### Currently have full models

Today we have so much deep learning models. Each model are quite good at some problem and cannot got 1 models good at all. Each task need to choose the best model:

> **Deep Work:** Professional activities performed in a state of distraction-free concentration that push your cognitive capabilities to their limit. These efforts create new value, improve your skill, and are hard to replicate.

Here are some possible examples of deep work (possibly biased towards my own profession and interests):

* Computer programming
* Playing a musical instrument
* Writing a book or paper
* Producing something complicated with your own hands (e.g. pottery or carpentry)

Okay, let’s also have a quick look at the antonym, just to make sure:

> **Shallow Work:** Non-cognitively demanding, logistical-style tasks, often performed while distracted. These efforts tend to not create much new value in the world and are easy to replicate.

* Surfing the Internet for information
* “_[…] [S]ending and receiving e-mail like a human network router[...]_”
* Formatting existing text and filling Excell tables

Here is the proposed shorthand to differentiate the two kinds of work in ambiguous cases. Simply ask yourself:

> How long would it take (in months) to train a smart recent college graduate student with no specialized training in my field to complete this task?

If the answers is _“one or two”_ it most likely isn’t that deep of an activity (though those month might require a lot of deep work from the student to get there).

### Why you should care

I think deep work has always been important and most impressive things have been created by people practicing it. Be it great literature, music, art or inventions (though, especially in science, a fair amount of luck as well, which is regularly overlooked). The growing necessity of deep work is, however, new. This is of course due to the _“shift to an information economy”_ where _“more and more of our population are knowledge workers”_. Therefore, Newport proclaims deep work to be “_the superpower of the 21st century_”. Here is his hypothesis:

> **The Deep Work Hypothesis:** The ability to perform deep work is becoming increasingly rare at exactly the same time it is becoming increasingly valuable in our economy. As a consequence, the who cultivate this skill, and then make it the core of their working life, will thrive.

There are, as mentioned in the beginning, several paths of reasoning to come to this conclusion.

#### 1. Economic reasons

To thrive in the current and future information economy you need at least the following two core abilities which both depend on your ability to perform deep work:

* The ability to quickly master hard things. “_If you can’t learn, you can’t thrive_”.
* The ability to produce at an elite level, in terms of both quality and speed.

> **The Intellectual Life** by _Antonin-Dalmace Sertillanges_: [Wo]Men of genius themselves were great only by bringing all their power to bear on the point on which they had decided to show their full measure.

You can formulate productivity as a scientific problem to then systematically solve it:
$$
\mathrm{High\ Quality\ Work\ Produced} = \mathrm{Time\ Spent} \times \mathrm{Intesity\ of\ Focus}
$$
As no one wants to waste time, your only other option to be more productive is to increase you intensity of focus. This, of course, is easier said than done. Interruptions are an especially fierce enemy of deep work, as it was found that your attention partly rests with the previous task when you switch, so you will never be able to “_bring all [your] power to bear on the point on which [you] had decided to show [your] full measure_”. So much for multi-tasking.

Only knowing this one fact allows you to critically observe many of your potential habits, like constantly checking your mail or social media, or your work setup, e.g. whether you work in an open office or not (which might create more opportunities for collaboration but do so at the cost of massive distraction). Another enemy of deep work:

> **The Principle of Least Resistance:** In a business setting, without clear feedback on the impact of various behaviors to the bottom line, we will tend toward behaviors that are easiest in the moment.

In other words, if there are no incentives to work deeply, you will prefer shallow work. Here is one possible explanation for this behavior:

> **Busyness as a Proxy for Productivity:** In the absence of clear indicators of what it means to be productive and valuable in their jobs, many knowledge workers turn back toward an industrial indicator of productivity: doing lots of stuff in a visible manner.

#### 2. The idle mind is the devil’s workshop

With the boring stuff out of the way, let’s now look a more interesting reason to favor deep over shallow work: It’s simply more fun. If you are anything like me, you know the satisfaction from doing something manual, with your own two hands. Be it the renovation of an old piece of furniture, repairing your bike, building something or baking bread. Those activities seem inherently meaningful. Why is it then, you might ask, I rarely get this feeling from my professional work? As it turns out, it might be, that you are simply not equally immersed into those tasks.

> Our brains […] construct our worldview based on what we pay attention to. Who you are, what you think, feel, and do, what you love—is the sum of what you focus on.

The task of a craftsman, and that’s you when doing something crafty, “_is not to generate meaning, but rather to cultivate in himself [or herself] the skill of discerning the meanings that are already there._” In other words, the meaning is already present in any demanding work, you just need to pay more attention at what you are doing instead of flicking back and forth between a million distracting whims. That can even be true for simple tasks though, as long as there is something to work towards.

> We who cut mere stones must always be envisioning cathedrals.

Luckily, we can exploit one useful property: Jobs are actually easier to enjoy than free time! This is because, as mentioned above, being in a state of flow, fully focusing on an interesting problem, is tremendously pleasing, and work is inherently flow-conducive as it features “_built-in goals, feedback rules and challenges, all of which encourage one to become involved in one’s work, to concentrate and lose oneself in it. Free time, on the other hand, is unstructured, and requires much greater effort to be shaped into something that can be enjoyed”_.

That’s certainly something I noticed when traveling alone through Canada after finishing my Bachelor’s degree, where each day I was entirely free to do whatever I liked. A state which quickly became quite burdensome, as all the responsibility of having a good day was entirely up to me and me alone. And of course I was expecting nothing less of myself than having the best time of my life. In the end I resolved this calamity by doing volunteer work, an activity that felt much more meaningful, and therefore more enjoyable, than traveling aimlessly from one town to the next.

## Part 2: Becoming a deep worker

> I have been a happy man ever since January 1, 1990, when I no longer had an email address. I’d used email since about 1975, and it seems to me that 15 years of email is plenty for one lifetime. Email is a wonderful thing for people whose role in life is to be on top of things. But not for me; my role is to be on the bottom of things. What I do takes long hours of studying and uninterruptible concentration.

You have diligently worked your way through the elevating prose or boldly skipped the first part of this article and are now eager to get something tangible? Either way, let’s cut to the chase: What follows are rules, tips and propositions to do better work and simultaneously have more fun doing so (click on the arrows to read more).

### Rule 1: Work deeply

<details markdown="1">
<summary><strong>You have a finite amount of willpower that becomes depleted as you use it</strong></summary>

You should therefore build yourself a daily routine to work by, which helps to spend less energy on organizing your work and more time on actually doing the work. This means you should schedule your deep work efforts. Consider using one of these options:

1. _The monastic philosophy of deep work scheduling:_ If your work is defined by one primary discrete goal and your success depends on doing this one thing exceptionally well–like writing books or articles if you are a writer–this might be for you.

     The idea is as radical as it is simple: While you work, eliminate _all_ influences that could interfere with your goal to get as much uninterrupted time as possible. This could mean going completely of the grid and even hiding from the world by going to a remote location—e.g. a small house on a Scottish island–for the time you need to complete you next project.

     > If I organize my life in such a way that I get lots of long, consecutive, uninterrupted time-chunks, I can write novels. But as those chunks get separated and fragmented, my productivity as a novelist drops spectacularly.

2. _The bimodal philosophy of deep work scheduling:_ If your work consists of tasks which require high levels of concentration for long consecutive time chunks but at the same time high degrees of interaction with world you might consider scheduling your deep work in this way. Among those employing this strategy might be academics, who need to write papers which push the boundary of what is currently known but at the same time have to collaborate with other researchers, attend conferences or give classes.

     To do so, divide your time into stretches of uninterrupted deep work and leave the rest of your time open for everything else. This could mean that, for some days, you refrain from checking your email or other media, retreat to a quiet place to work and concentrate on your complicated tasks until you are satisfied, then reemerge. This division of time can happen on multiple scales such that you dedicate four days a week or one week a month, or one season of a year to your deep work. It should be compatible with a large proportion of knowledge worker jobs but might require some adjustments in coworker expectations and clear communication.

3. _The rhythmic philosophy of deep work scheduling:_ If you can’t even disconnect for one or two days (and remember, we are only talking about your work life here), this might be the way to go. The only thing asked from you here is consistency. Dedicate a certain proportion of your day to deep work and then stick to it. _Every day_. The idea is that your form a habit of working deeply such that you don’t need to think about setting it up. You could probably also use this strategy as an entry level effort to work deeply and see where it gets you as it is also likely the most straight forward method of incorporating deep work into your normal schedule.

     It the philosophy I’ve chosen for my own work. I’ll stay disconnected in the morning, focus on complicated work till lunch and only then go online to check my email and indulge in other shallow activities.

4. _The journalistic philosophy of deep work scheduling:_ Finally, if you can’t actually schedule your deep work reliably in any other way, just switch into deep work mode immediately whenever an opportunity pops up! This comes with a big caveat though: “_just switch into deep work mode immediately_” is just not an option for most people. As discussed [earlier](#1-economic-reasons), efficiently switching between tasks is neurologically problematic and will deplete your finite willpower reserve. If possible at all, it requires a lot of training, so this strategy is definitely not recommended for the deep work novice. If you want to give it a try, at least try to think about possible deep work windows in advance on e.g. a weekly basis.
</details>

<details markdown="1"><a name="dont-work-from-inspiration"></a>
<summary><strong>Don’t work from inspiration</strong></summary>

> There is a popular notion that artists work from inspiration---that there is some strike or bolt or bubbling up of creative mojo from who knows where... but I hope [my work] makes clear that waiting for inspiration to strike is a terrible, terrible plan. In fact, perhaps the single best piece of advice I can offer to anyone trying to do creative work is to ignore inspiration.

As it turns out, most important thinkers and great minds didn't wait for inspiration to strike but instead employed rituals designed to actively encourage their brains to be productive.

The argument is in the same vain as used above: ritualizing, i.e. forming a habit, helps to minimize the friction in the transition to depth. There is not one correct deep work ritual, but some general points should nonetheless be addressed:

1. **Where you'll work and for how long:** We did already discuss the _when_ above: Regardless of where you work, set yourself a specific time frame to keep the session a discrete challenge and not an open ended slog. The _where_ should be a quiet, tidy place; even better if used exclusively for deep work.
2. **How you'll work once you start:** Give yourself rules extending the basic deep work definition. This might be a ban of Internet use or a metric like words produced per twenty minutes. Without clear rules and goals you'll waste energy and willpower on thinking about how to work while you work.
3. **How you'll support your work:** How well you can focus doesn't only depend on your environment, but also on other factors like how you feel. You might want to have a ritual to settle into your deep work like a good cup of coffee before you start or some meditation, yoga or exercise. This might also include organizing the raw materials, like paper and pen, before starting.

Finding your personal deep work habit might take some time and experimentation, but is definitely worth it, because the more you use it, the more it will rewire your brain such that simply following your ritual brings you into a state of intense concentration.
</details>

<details markdown="1">
<summary><strong>Make grand gestures</strong></summary>

By leveraging a radical change to your normal environment, coupled perhaps with a significant investment of effort or money, all dedicated toward supporting a deep work task, you increase the perceived importance of the task. This boost in importance reduces your mind's instinct to procrastinate and delivers an injection of motivation and energy.

This somehow mimics the effect an external deadline or exam date has on most people, but, because it is created by you artificially, can be utilized whenever you need it.
</details>

<details markdown="1">
<summary><strong>Don’t work alone</strong></summary>

While the relationship between deep work and collaboration is tricky, it is often still worth trying to work out a problem with a partner. If you’ve ever discussed various approaches to a problem in front of a black or white board in preparation of a tough exam you know what this is about.

The key for this to work is often the ability to work deeply on the problem on your own first and _then_ discuss it with someone else. This is also why open offices often degrade productivity but the ability to meet people _in between_ work, e.g. during a coffee or lunch break, is extremely helpful.
</details>

<details markdown="1">
<summary><strong>Execute like a business</strong></summary>

> You are such a naive academic. I asked you _how_ to do it, and you told me _what_ I should do. _I know what I need to do. I just don’t know how to do it._

The division between _what_ and _how_ is crucial but is often overlooked. It’s often much easier to identify a strategy for achieving a goal than figuring out how to actually execute it once identified.

Say you want to become a great climber. Well, that’s easy! Just go climbing a lot and do lots of training. The difficult part is how to incorporate this into your life so that you have the time and motivation to pull through with it.

Enter _The 4 Disciplines of Execution_:
1. **Focus on the wildly important:** The more you try to do at once, the less you will accomplish. You should therefore concentrate only on the wildly important goals. In our case this is to work deeply. But because the general credo to spend more time working deeply doesn’t spark a lot of enthusiasm, you are better of setting yourself a specific, tangible goal like publishing five high-quality peer-reviewed papers in one year if you’re an academic.

   > If you want to win the war for attention, don’t try to say ‘no’ to the trivial distractions you find on the information smorgasbord; try to say ‘yes’ to the subject that arouses a terrifying longing, and let the terrifying longing crowd out everything else.

2. **Act on the lead measures:** Okay, so you have identified your wildly important goal, but now you need to measure your success. There are two ways to look at this. (1) Have I achieved my goal? (2) Am I on the right track to achieving my goal i.e. is my environment and behavior conducive to making progress towards my goal?

   Is one of these angles superior? It turns out, it is. We might call the first approach the _goal_ measure and the second one the _process_ measure (they are sometimes also called _lag_ and _lead_ measure). The problem with the first approach is, that the feedback comes too late to change your behavior. Once the year is over and you look at your performance, it is already too late to change anything. The goal measure lacks impact on your day-to-day behavior.

   The second approach measures your success indirectly, but constantly, so you can tweak your behavior if needed. A good example for the process measure might be the time you spend each day working deeply. The more it is, the more you probably get done in a day, which has an indirect positive effect on you actual goal.

3. **Keep a compelling scoreboard:** I think of this discipline as a way of _gameification_. You set up some visual way of recording your achievements such that it is immediately visible how well you’re doing and that it would feel terrible to destroy your two weeks run of daily deep work or that you feel compelled to keep increasing this deep work counter.

4. **Create a cadence of accountability:** Keep yourself accountable. Plan your day (and if possible week) ahead of time and record your progress. At the end of the day (or week), review your work, celebrating when you did well ([I’m proud of you son](https://www.youtube.com/watch?v=M1B3gATS0GE)) and analyzing what went wrong so you can adapt your plan for the next day(s).
</details>

<details markdown="1">
<summary><strong>Be lazy</strong></summary>

> Idleness is not just a vacation, an indulgence or a vice; it is as indispensable to the brain as vitamin D is to the body, and deprived of it we suffer a mental affliction as disfiguring as rickets… it is, paradoxically, necessary to getting any work done.

At the end of the workday, shut down your consideration of work issues until the next morning—no after-dinner email check, no mental replays of conversations, and no scheming about how you’ll handle an upcoming challenge; shut down work thinking completely.

* **Downtime aids insight:** Interestingly, some decisions are better left to your unconscious mind (have a look at _Thinking Fast and Slow_ by Daniel Kahneman, a fantastic book on behavioral psychology, if you want to know more). You’re probably familiar with the lightning bold of insight that strikes you unprepared in the middle of the night, or while running or while commuting but usually not at precisely the moment when you think about the problem the most. So just sit back and let your brain sift through and untangle some of the information for you.

* **Downtime helps recharge the energy needed to work deeply:** Your capability to concentrate and pay attention is a finite resource. You need to give it time to recharge. This works best by going outside and getting some fresh air, sun and exercise. No surprises here. A park or forest is especially well suited, as it is simple enough to navigate (no streets, cars and traffic lights) so that you don’t further deplete your attention reserves but at the same time intriguing enough to distract you from returning to your problems.

  > “Only ideas won by walking have any value.” – Friedrich Nietzsche

* **The work that evening downtime replaces is usually not that important:** A novice deep worker can muster somewhere around an hour a day of real, intense concentration while an expert can extend this to about four, but rarely more (if that seems strange to you, check out [this paper](https://pdfs.semanticscholar.org/f6f1/d52a73ace9361b0a16363bd5481ffa920c7b.pdf) on the topic). So provided you don’t start your day at 6 pm, the probability that you accomplish something truly valuable in the evening is slim.
</details>

<details markdown="1">
<summary><strong>Shut it down!</strong></summary>

Sometimes its hard to let go of problem, even, or especially if you have been working on it all day without making much progress. We already discussed the importance of being lazy and the marginal likelihood that you will do anything of value at the end of your work day. To help you to disconnect, try to conceive of a shutdown ritual.

It should ensure that every incomplete task, goal or project has been reviewed and that for each you have confirmed that either (1) you have a plan you trust for its completion, or (2) it’s captured in a place where it will be revisited when the time is right. The process should be an algorithm: a series of steps you always conduct, one after another. Here is an example:

1. Check your email on last time to ensure there is nothing that needs an urgent response right away.
2. Put all open tasks and project into your todo list.
3. Check all tasks and your calendar for the next few days to ensure you didn’t overlook anything vital.
4. Use this information to make a rough plan for the next day.

When you’re done, have a set phrase you say that indicates completion. Especially the last step might sound weird, but its a mental cue, telling your brain that it’s safe to let go. It could be as simple as “_I’m done for today!_” or the more eccentric “_Shutdown complete_”.

If you are not convinced, read up on the [Zeigarnik effect](https://en.wikipedia.org/wiki/Zeigarnik_effect) (though apparently there is some controversy around the validity of it, I have noticed for myself to have trouble letting go of any unfinished task as long as it isn’t recorded somewhere).
</details>

### Rule 2: Embrace boredom

> The ability to concentrate intensely is a skill that must be trained.

While this might sound obvious once pointed out, it is common to treat undistracted concentration like a _habit_ that you have just forgotten to pick up on due to lack of motivation but that you can simply switch on if really needed.

There is further an important corollary to this idea of strengthening your mental muscle: Efforts to deepen your focus will struggle if you don’t simultaneously wean your mind from a dependence on distraction—fleeing from the slightest hint of boredom—just like an hypothetical athlete, thwarting all positive effects of training by indulging in mindless eating frenzies in between.

To put this more concretely: If every moment of potential boredom in your life—say, having to wait five minutes in line or sit alone in a restaurant until a friend arrives—is relieved with a quick glance at your smartphone, then your brain has likely been rewired to a point where it’s not ready for deep work—even if you regularly schedule time to practice this concentration.

<details markdown="1">
<summary><strong>Schedule your use of the Internet</strong></summary>

Instead of scheduling the occasional break _from distraction_ so you can focus, you should instead schedule the occasional break break _from focus_ to give in to distraction.

Schedule in advance when you’ll use the Internet, and then avoid it altogether outside these times. Of course, this requires some kind of planning beforehand to not get stuck in your work before your scheduled reconnect occurs.

The problem here is not the Internet itself, but the constant _switching_ between low-stimuli/high-value activities (deep work) to high-stimuli/low-value activities (shallow work).

* If you’re required to spend hours every day online or answer emails quickly, that’s fine: It simply means that your _Internet blocks_ will be more numerous.
* It doesn’t matter how you schedule your use of the Internet, but once you do, stick to your plan. Try switching to another offline activity when stuck (or even just relax).
* Also scheduling your Internet use at home can further improve your concentration training.
</details>

<details markdown="1">
<summary><strong>Give yourself deadlines</strong></summary>

Identify a deep task (that is, something that requires deep work to complete) that’s high on your priority list. Estimate how long you’d normally put aside for an obligation of this type, then give yourself a hard deadline that _drastically reduces_ this time.

If possible, commit publicly to the deadline—for example by telling the person expecting the finished project when they should expect it. If this isn’t possible, motivate yourself by setting a timer.

At this point, there should be only one possible way to get the deep task done in time: _working with great intensity and without any distraction_. A beneficial side effect of this practice is, that you will greatly improve your ability to estimate the time you need to finish any given task.
</details>

<details markdown="1">
<summary><strong>Productive meditation</strong></summary>

Let me begin this section by pointing out that the notion of “_productive meditation_” is somewhat orthogonal to the practice of actual meditation. While, contrary to popular belief, the “_goal_” of meditation is not to stop thinking, it also certainly isn’t to be fully distracted by thoughts and _“to get things done”_. This probably comes down to the difference between _complete distraction_ and _complete focus_, which, depending which way you look at it, seem to be two remarkably similar states.

Anyway, the goal of productive meditation is to take a period in which you’re occupied physically but not mentally—walking, jogging, driving, showering—and focus your attention on a single well-defined professional problem.

As a mindfulness meditation, you must continue bring your attention back to the problem at hand when it wanders or stalls.

This practice not only helps to get things done, but also trains your mind to think deeply, sharpening your concentration and strengthening your distraction-resistance muscle.

* Just like with any form of meditation, be wary of distractions and looping (thinking the same thoughts over and over again, avoiding the strainous activity to actually make progress).
* Structure your deep thinking by first identifying the relevant _variables_ for solving the problem, (e.g. the main points you want to make in your next book chapter or the actual variables you need for a mathematical proof) then define a specific next-step question to work on (e.g. _“how to best open the next chapter?”_, “_what happens if I assume that this proposition holds?_”).
</details>

<details markdown="1">
<summary><strong>Memorize a shuffled deck of cards</strong></summary>

I think of myself having a particularly bad memory so this idea scares me like crazy. Which probably means I should do it.

So if you decide to do it, here are two obvious questions:

1. **Why?:** It was found that one of the biggest differences between memory athletes and the rest of us is in a cognitive ability that’s not a direct measure of memory at all _but of attention._ Does this ring a bell? Exactly, apparently we can train our memory by training to pay more attention which in turn helps to work with greater concentration.

2. **How?:** Professional memory athletes _never_ attempt to memorize by looking at the information again and again. Incidentally, this is probably the way you (and me) usually try to get stuff into our brain. The problem with this approach is, that it misunderstands how our brain works. We are not wired to quickly internalize abstract information but instead we are really good at remembering scenes. What follows is _one_ possible way to make use of this insight to memorize a large number of abstract things:

   1. Start by imagining a walk through five distinct rooms (or areas) in your home, one after another, keeping the order fixed.

   2. Select 10 items in each room which define its purpose, like the fridge and stove in the kitchen. The items should be large. Add an additional 2 arbitrary items to get to the 52 needed to memorize an entire deck of cards.

   3. Establish an order in which you look at these items (e.g. when coming into the kitchen, you first look at the fridge, then at the stove…)

   4. Practice the mental exercise of walking through your home and looking at the objects. This should feel much easier than doing so with 52 random items, but once you can do it, you have already successfully memorized 52 random things!

   5. (Optional): Associate each of the 52 cards with a person or thing. The association should feel somewhat intuitive or logical like Queen Elizabeth with the queen of hearts or Donald Trump with the king of diamonds. I find this step somewhat difficult, because I’m not good at memorizing people and it feels like I need to memorize twice as much.

      Practice these associations until you can immediately recall the person for a randomly chosen card.

   6. Final step: Begin your walk-through of your home. As you encounter each item, look at the next card from the shuffled deck, and imagine the corresponding memorable person or thing doing something memorable near or with that item. This might be Donald Trump watching some Fox News on your TV in the living room if the current card is the king of diamonds.

   The cool thing is, that you need to do steps 1 to 5 only once and can then make use of them whenever you need. It still feels super difficult in my opinion, but I’ll give it a try soon.
</details>

### Rule 3: Quit social media

> I’ll pay attention to you what you say if you pay attention to what I say—regardless of its value.

Most people have recognized by now, that network tools, like social media, fragment their time and reduce their ability to concentrate[^1]—a notion confirmed by a large and growing corpus of research on the subject—and that this can be a cause of distress and even depression.

[^1]: Among other problems like anxiety to miss out and due to constantly comparing yourself to others as well as data privacy concerns.

On the other hand, those tools are neither inherently evil nor inherently benevolent. The question is therefore not whether to use them or not, but rather which to use and how much time to devote to them.

<details markdown="1"><a name="how-to-approach-network-tools"></a>
<summary><strong>How to approach network tools</strong></summary>

You might get defensive when reading a title like _“quit social media”_ or even when asked to merely tone done your usage a little. Your arguments might be along the line of “_it helps me to stay connected to my friends_” or “_it helps me to discover new groups and events I otherwise wouldn’t know about_” and they are valid; for someone new to a city, job or university. But once you’ve settled in and identified that your work is also important to you, they seem to be rather superficial.

It’s not like you wouldn’t have other entertainment options if deprived of social media. And are those friends you merely meet online really that important to you? In other words, are they central to your social life? How does chatting with someone compare to actually meeting this person? It’s of course up to you to answer those questions for yourself, but try not to fall into the following trap:

> **The Any-Benefit Approach to Network Tool Selection:** You’re justified in using a network tool if you can identify any possible benefit to its use, or anything you might possibly miss out on if you don’t use it.

This line of thought completely ignores the potential downsides of the given tool, which, as discussed before, can be severe. It is also interesting to contrast this approach with another, utilized by any skilled laborer throughout history:

> **The Craftsman Approach to Tool Selection:** Identify the core factors that determine success and happiness in your professional and personal life. Adopt a tool only if its positive impacts on these factors substantially outweigh its negative impacts.

Can you imagine buying a new phone, tv, drill or or bike without thinking about the downsides but by simply relying on the heuristic: _“It could be of some use to me, so I must have it.”_—No. So why not apply this approach to digital tools as well?
</details>

<details markdown="1">
<summary><strong>The law of the vital few</strong></summary>

Here is how you could go about sifting through your bag of tools:

1. **Identify the main high-level goals in both your professional and personal life:** These could involve being an effective researcher and a good teacher and mentor if you are a professor, writing great stories if you are a writer, as well as spending a lot of time with your friends, partner, kids or family.

2. **List for each of those goals the three most important activities:** This could be “_regularly read and understand the cutting-edge results in my field_” and “_finish work at 4 pm so I have the rest of the day for my friends and family”_.

3. **Go through your list of network tools.** Ask yourself whether it has a _substantially positive_, _substantially negative_ or little impact on your regular and successful participation in these activities.

4. **Finally, _only keep using the tool if it has a substantially positive impact which outweighs any negative ones_.** You can of course go back and adjust your goals, but once you are certain about what you want, there is simply no good reason to keep stuff around that’s not actively contributing to it or even hurts it.

In doing so, you make use of

> **The law of the vital few:** In many settings, 80 percent of a given effect is due to just 20 percent of the possible causes.

in that probably only around $20\%$ of your activities are responsible for $80\%$ of your wellbeing and a mere $20\%$ of the network tools you are using makes up for $80\%$ of the positive impact on succeeding in these activities.

As your life is a zero-sum game, time invested in one activity will be subtracted from the time invested in all others, it makes sense to focus all your time and attention on those activities and tools that provide the greatest benefit.
</details>

<details markdown="1">
<summary><strong>Getting rid of unnecessary things</strong></summary>

Here is an idea to muck out all the unnecessary stuff once and for all: Pack it all up like you are going to move, then spend a normal week, only unpacking whatever you need and put it back where it used to go. At the end of the week, get rid of whatever is left in the boxes.

Admittedly, this is a rather radical approach, but we can translate it into a way of getting rid of unnecessary network tools. Instead of packing them, don’t use them for a month. All of them. Don’t deactivate them, and don’t tell anyone[^2]. Just stop using them. Explain what you are doing if necessary (e.g. someone reaches out to you by other means because (s)he worries about what happened to you). Then, ask yourself the following questions:

[^2]: It is part of the seductive quality of social media that they make you think that people _want to hear what you have to say_. You’ll soon find out.

1. Would the last 30 days have been _notably_ better if I had been able to use this service?
2. Did people care that I wasn’t using this service?

If your answer is “no” to both questions, quit the service permanently.

</details>

<details markdown="1">
<summary><strong>Don't use the Internet to entertain yourself</strong></summary>

The quintessence of this idea is: _"Put more thought into your leisure time"_. This simply means, that you can and should actively decide what you want to do once you’ve stopped working instead of getting captured by whatever catches your attention at this moment. Addictive websites are just one of those attention traps.

If you’ve _planned_ to watch the newest video of your favorite YouTuber, that’s of course completely fine. Just try not to get drawn into the maelstrom. As mentioned before, work can be surprisingly pleasing due to its inherent feedback and reward cycles. In planning your free time as well, you can transfer some of those benefits and you will end your day feeling more fulfilled.

</details>

### Rule 4: Drain the shallows

> Treat shallow work with suspicion because its damage is often vastly underestimated and its importance vastly overestimated.

<details markdown="1">
<summary><strong>Account for every minute of your workday</strong></summary>

While this might at first sound overly bureaucratic, the idea is similar to mindfulness meditation: Pay attention to what you are doing instead of relying on your autopilot who is usually pretty lenient towards the trivial creeping in.


If, like me, you are not a mindfulness master yet, it helps to make a concrete plan for the day. This one can be a lot more detailed than the simple allocation of tasks you did the night before. If you don’t know exactly what you will be doing at, say, 2 pm, make your best guess or use a broad label like _“administrative tasks”_.

The important thing here is not that you manage to follow your plan to the minute—you can always revise it if needed—but that planning in this way lends some structure to your day and prevents you from wasting precious motivation on constantly figuring out what to do next. Treat your time with respect.

Again, the goal of the a schedule is not to force you into a rigid plan but rather to make you think about what you actually want. It’s not about _constraint_ but about _thoughtfulness_. And you might even be _more_ creative than someone employing the more traditional “_spontaneous_” approach ([Don’t work from inspiration](#dont-work-from-inspiration))
</details>

<details markdown="1">
<summary><strong>What percentage of my time should be spent on shallow work?</strong></summary>

If you work for someone else, ask him or her this question, if you work for yourself, ask _yourself_. Whatever number you settle on, _try to stick to this budget_.

If you have no clue, start from the extremes and work yourself towards a reasonable answer. $0\%$? No. $100\%$? No. $50\%$? Still a bit much, you didn’t spend that much time studying or in training to waste half your day on easy to replicate, low-value stuff, right? You get the gist.

Be aware though, that this will most likely require you to change your habits around taking on work infused with shallowness and your availability.

Remember, try to avoid falling into the [any benefit trap](#how-to-approach-network-tools).
</details>

<details markdown="1">
<summary><strong>Fixed-schedule productivity</strong></summary>

> Deliberately do specific things to preserve your happiness.

Give yourself a fixed deadline for when to stop working and stick to it. This not only helps to make sure you have enough time for other activities and to preserve and replenish your energy to work deeply but further motivates you to actually focus on your work and avoid distractions.

For the above suggestion to work you'll need to prioritize certain tasks and turn down others. Being more productive by being more focused has it's limits so don't blindly take on board anything that comes floating by.

> Be incredibly cautious about your use of the most dangerous word in one's productivity vocabulary: "yes".
</details>

<details markdown="1">
<summary><strong>Become hard to reach</strong></summary>

The biggest burden on your concentration is probably your constant availability. We have already discussed social media, so here are some tips specifically targeted at email.

* **Tip 1: Make people who send you email do more work.** If you have a public place advertising your email address and you get bombarded with messages asking for advice, adding a _sender filter_ can help a lot. Simply put a small text around your address, explaining what sort of message you would like to receive and also what sort of message will likely end up being answered by you. This way, the burden is on the sender to convince the receiver that a reply is worthwhile.

* **Tip 2: Do more work when you send or reply to emails.** You know those two line emails along the line of: _“I took a stab at that article we discussed. It’s attached. Thoughts?_ This took the “author” a few seconds to draft and certainly must have felt good—one more email dealt with—but potentially asks hours of your time in return. _Don’t be that person_. Instead, think about how you make life easier for your correspondence, for example by sharing a summary of this article in question, highlighting certain points and giving your own opinion in advance.

  > **The process-centric approach to email:** What is the project represented by this message, and what is the most efficient (in terms of messages generated) process for bringing this project to a successful conclusion?

* **Tip 3: Don’t respond.** Again: _It’s the sender’s responsibility to convince the receiver that a reply is worthwhile_. If it’s not, don’t.
</details>

### Closing thoughts

And there you have it. I hope you could take away one or two useful crumbs to incorporate into your working habits. A lot might have felt a little over boarding and rigid, at least it did for me from time to time, but maybe just give it a try and see where it gets you. Before you reject the idea outright, give this final consideration a spin through your neuronal wiring:

> There’s […] an uneasiness that surrounds any effort to produce the best things you’re capable of producing, as this forces you to confront the possibility that your best is not (yet) that good.
</div>

## Deep Work: A 5 minutes blitz

The topic interests you but an entire book is overkill and even this article feels like a big commitment? I don’t blame you. Instead, here is a 5 minutes crash course to deep work that you can simply fill out, <a onclick="window.print(); return false;">**print**</a> and pin next to your desk. There will certainly be rules and tips that sound strange without context so feel free to [read up](#part-2-becoming-a-deep-worker) on anything unclear.

<input type="text" placeholder="Fill in your own text here"> and commit to an idea by ticking the <input type="checkbox"> next to it.

### What is _Deep Work_?

> **Deep Work:** Professional activities performed in a state of distraction-free concentration that push your cognitive capabilities to their limit. These efforts create new value, improve your skill, and are hard to replicate.

### What is _not_ deep work?

> **Shallow Work:** Non-cognitively demanding, logistical-style tasks, often performed while distracted. These efforts tend to not create much new value in the world and are easy to replicate.

### How do you want to work?

For your deep work, you go completely of the grid 
1. <input type="checkbox"> _Monastic:_ for extended periods of time (i.e. several weeks/month).
2. <input type="checkbox"> _Bimodal:_ for at least a day at a time.
3. <input type="checkbox"> _Rhythmic:_ for at least an hour a day with a fixed schedule.
4. <input type="checkbox"> _Journalistic:_ at every possible opportunity (not for beginners).
5. <input type="text" class="mid" placeholder="Chill: I'll do deep work once in a full moon.">

### When do you want to work?

Regardless of what you decided above, choose a specific timeslot for you deep work:<br>
<input type="text" class="mid" placeholder="I'll do my deep work between 7 am and 11 am everyday.">

### Where do you want to work?

It should be a quiet, tidy place; even better if used exclusively for deep work.<br>
<input type="text" class="mid" placeholder="I'll do my deep work on my desk.">

### Which rules do you follow?

1. <input type="text" class="mid" placeholder="I'm not going to use the Internet once I've started.">
2. <input type="text" class="mid" placeholder="I evaluate progress every 20 minutes.">
3. <input type="text" class="mid" placeholder="I do 2 sessions of 90 minutes each.">

### How will you support your work?

1. <input type="text" class="mid" placeholder="A good cup of coffee before I start.">
2. <input type="text" class="mid" placeholder="A long break after 90 minutes.">
3. <input type="text" class="mid" placeholder="Some excersise when I'm done.">

### What's your emergency plan?

To boost your motivation on this wildly important task:<br>
<input type="text" class="mid" placeholder="I will treat myself to a new bicycle if I finish this by tomorrow night!">

### It's not about _what_ you need to do but _how_ to do it.

1. I'll focus on the following wildly important goal:<br>
	<input type="text" class="mid" placeholder="Find a topic for my PhD thesis.">
2. Ill use the following strategy to get there:<br>
	<input type="text" class="mid" placeholder="Read at least one paper or blog post on an interesting topic each day.">
3. I'll track my success each day using:<br>
	<input type="text" class="mid" placeholder="my everyday calendar.">
	<a href="https://www.youtube.com/watch?v=-lpvy-xkSNA&t=1s" style="padding: 0 6px; border: 1px solid; border-radius: 50%;">?</a>
4. I'll keep myself accountable by:<br>
	<input type="text" class="mid" placeholder="planning my week ahead of time and reviewing my progress when it's over.">

### What's your shutdown ritual?
1. <input type="text" class="mid" placeholder="I check my email one last time for anything urgent">
2. <input type="text" class="mid" placeholder="I put all unfinished work into my todo list.">
3. <input type="text" class="mid" placeholder="I check my todo list and calendar for things I might have overlooked.">
4. <input type="text" class="mid" placeholder="I make a rough plan for the next day.">
5. <input type="text" class="mid" placeholder="I give my brain a cue to let go by saying 'I'm done for the day!'">

### How will you schedule your use of the Internet?
<input type="text" class="mid" placeholder="I'll only make use of the Internet in the afternoon when my deep work is done.">

### Comitments:
* #### I promise not to think about work once I've stopped. <input type="checkbox">
  
* #### I'll give myself a competitive deadline for all of my tasks. <input type="checkbox">

* #### I'll use the time commuting to make progress on a specific problem. <input type="checkbox">

* #### I'll learn to memorize a shuffled deck of cards to train my memory. <input type="checkbox">

* #### I'll make use of the _Craftsman Approach to Network Tool Selection_ <input type="checkbox">
  <details markdown="1">
  <summary>Sure, but what's that?</summary>

  > **The Craftsman Approach to Tool Selection:** Identify the core factors that determine success and happiness in your professional and personal life. Adopt a tool only if its positive impacts on these factors substantially outweigh its negative impacts.

  1. _What is the main high-level goal in your professional/personal life?_
     * Prof.: <input type="text" class="mid" placeholder="To be an efficient and effective researcher.">
     * Personal: <input type="text" class="mid" placeholder="To spend as much time as possible on my hobbies and with my friends/partner.">
  2. _What are the three most important activities to succeed with each goal?_
     * Professional:
     1. <input type="text" class="mid" placeholder="Read/understand important publications in my field.">
     2. <input type="text" class="mid" placeholder="Discuss/collaborate with other researchers.">
     3. <input type="text" class="mid" placeholder="Publish high-quality peer-reviewed papers.">
     * Personal:
     1. <input type="text" class="mid" placeholder="Quit work at 4 pm.">
     2. <input type="text" class="mid" placeholder="Don’t get drawn into addictive websites/apps.">
     3. <input type="text" class="mid" placeholder="Go to bed early so I can get up early where I’m most productive.">
  3. _Ask yourself for each network tool you're using: Which has a substantially positive/negative or neutral effect on those activities?_
  4. _Only keep using the tool if it has a substantially positive impact which outweighs any negative ones._
  </details>
  
* #### I'll stop using all of social media for 30 days. <input type="checkbox">
  <details markdown="1">
  <summary>What?!</summary>

  Then ask yourself:

  1. Would the last 30 days have been _notably_ better if I had been able to use this service?
  2. Did people care that I wasn’t using this service?

  Quit each service for which the answer to both questions is "no".
  </details>

* #### I'll put more thought into my leisure time. (don't do what's easiest right now) <input type="checkbox">

* #### At the end of each day and each week I'll plan the following one. <input type="checkbox">

* #### What's your shallow work budget each day? <input type="text" style="width: 3em;" placeholder="30%">

* #### I'll stop working at <input type="text" style="width: 3em;" placeholder="4 pm"> each day.

* #### I'll be more mindful of other peoples time and put more work into my emails. <input type="checkbox">

Congratulations, you are on the way of becoming a real deep worker! If you like, <a onclick="window.print(); return false;">**print**</a> this overview[^3] to remind yourself.

[^3]: It will **not** be saved!

---
