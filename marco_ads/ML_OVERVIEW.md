# Reconstructing a Collapsing Shell from Entanglement Data: An Overview

## The goal

We consider a three-dimensional AdS spacetime in which a shell of matter collapses into a black hole (AdS$_3$-Vaidya):

$$
ds^2 = -f(r,v)\,dv^2 + 2\,dv\,dr + r^2 dx^2,\qquad f(r,v) = r^2 - m(v).
$$

Everything about the collapse is encoded in a single function, the mass profile $m(v)$, which rises from $0$ (empty AdS) to a final black-hole mass as the advanced time $v$ increases.

The question is: **can we recover $m(v)$ from boundary entanglement data alone?** For a boundary interval of width $\ell$ at boundary time $t$, holography relates the entanglement entropy to the length $L(\ell, t)$ of the spacelike geodesic in the bulk that connects the two ends of the interval. The data we are given is this function $L(\ell, t)$ (after subtracting the usual UV divergence), and nothing else.

## The idea in one sentence

We represent the unknown $m(v)$ by a small neural network $m_\theta(v)$, compute the geodesics this candidate spacetime would produce, compare their lengths with the data, and adjust $\theta$ by gradient descent, **differentiating through the geodesic computation itself.**

The neural network is not trained in the usual "input → label" way. It never sees $(\ell, t, L)$ triples as examples. It only defines a metric, and a physics solver turns that metric into predictions. All of the learning signal comes from asking whether the geodesics of the candidate spacetime reproduce the observed lengths.

## Step 1: The data

To test the method, we generate synthetic data from a known profile $m(v)$. We compute many boundary-anchored geodesics in that spacetime, record for each one where its endpoints sit on the boundary $(\ell, t)$ and how long it is, and interpolate the results onto a regular grid. The result is a smooth surface $L_{\text{data}}(\ell, t)$ that can be evaluated (and differentiated) at any point inside the covered region.

The true profile is then set aside. It is used only at the very end, to judge how good the reconstruction is.

## Step 2: Shooting geodesics through the candidate spacetime

Given the current network $m_\theta(v)$, we compute a family of geodesics in the metric $f = r^2 - m_\theta(v)$.

Each geodesic is symmetric about its deepest point in the bulk, the **turning point** $(r_\ast, v_\ast)$. We start there, with the geodesic momentarily moving only in the $x$ direction, and integrate the geodesic equations outward toward the boundary:

$$
\ddot v = r\dot x^2 - \tfrac12 \partial_r f\,\dot v^2,\qquad
\ddot r = f\,\ddot v + \partial_r f\,\dot r\dot v + \tfrac12\partial_v f\,\dot v^2,\qquad
\ddot x = -\tfrac{2}{r}\dot r\dot x .
$$

The term $\partial_v f = -m_\theta'(v)$ is the derivative of the network, obtained by automatic differentiation. The geodesics therefore respond both to the value of the mass and to how quickly it changes.

We do this for a whole grid of turning points, spread over a range of depths $r_\ast$ and times $v_\ast$. That is a few hundred geodesics, all integrated in parallel with a standard fourth-order Runge–Kutta scheme. Using many depths *and* many times is essential: geodesics at a single time do not contain enough information to pin down the profile.

## Step 3: Reading off boundary observables

Each geodesic eventually reaches a large cutoff radius $r_{\text{cut}}$ near the boundary. There we read off three numbers:

- the interval width $\ell$ (twice the $x$-distance travelled, since the geodesic is symmetric),
- the boundary time $t$ at which it arrives,
- its regularized length $L = L_{\text{proper}} - 2\log(2 r_{\text{cut}})$.

A subtlety is that "the step at which the geodesic crosses $r_{\text{cut}}$" is a discrete choice. Its derivative with respect to the network weights is zero almost everywhere, so gradients would not flow. We therefore replace the hard cutoff by a smooth one: a sigmoid weighting that switches off gradually around $r_{\text{cut}}$. This makes $\ell$, $t$ and $L$ smooth functions of the network weights, while reproducing the hard-cutoff values to high precision.

## Step 4: The loss

Here is the central trick. A geodesic launched from a given turning point in the *candidate* spacetime lands at some boundary point $(\ell_{\text{pred}}, t_{\text{pred}})$. That point depends on $\theta$ and generally differs from where the same turning point would land in the true spacetime. Rather than forcing it to hit a prescribed boundary point, which would require an expensive search inside every training step, we simply ask the data what the length *should* be at the point where the geodesic actually landed:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{k=1}^{N}
\Big[\, L_{\text{pred},k}(\theta) - L_{\text{data}}\big(\ell_{\text{pred},k}(\theta),\, t_{\text{pred},k}(\theta)\big) \Big]^2 .
$$

If the candidate profile equals the true one, every geodesic of the candidate spacetime is a genuine geodesic of the true spacetime, so its $(\ell, t, L)$ lies exactly on the data surface and the loss vanishes. The turning points are only a convenient way of labelling the candidate's own geodesics. The comparison itself happens purely in boundary quantities, so no bulk information leaks into the fit.

The gradient of this loss reaches $\theta$ along two routes: through the predicted length $L_{\text{pred}}$, and through *where* the geodesic lands (via the slope of the data surface at that point). Both are kept.

## Step 5: Training

- **Initialization.** The network is first briefly fitted to a deliberately wrong but physically sensible guess (a smooth step at the wrong place, with the wrong width). This gives well-behaved geodesics from the start without giving away the answer.
- **Optimization.** Each training step integrates all geodesics forward, evaluates the loss, and backpropagates through every one of the thousands of Runge–Kutta steps to get the gradient with respect to the network weights. We then take an Adam step with a slowly decaying learning rate. Typically a few hundred steps are used.
- **Memory.** Backpropagating through long integrations is memory-hungry. We store only the state at each integration step and recompute the intermediate Runge–Kutta stages during the backward pass (checkpointing).
- **Precision.** Everything runs in double precision, since the geodesics travel out to very large radii.

A training step takes a few seconds, and a full reconstruction runs in roughly half an hour.

## Step 6: Evaluation

After training, we compare $m_\theta(v)$ with the true profile, but only over the range of times that the geodesics actually probed. Outside that window the data carry no information about $m(v)$, so no method could be expected to get it right there. The figure of merit is the largest deviation $\max_v |m_\theta(v) - m_{\text{true}}(v)|$ within that window.

## What we have found so far

- **Single collapsing shells are recovered well.** Starting from a wrong initial guess, the reconstruction converges to the true profile, with errors of a few percent of the final mass.
- **The data have blind spots.** The fit can match the boundary data almost perfectly while still being noticeably wrong somewhere in the middle of the transition. Different mass profiles can produce nearly indistinguishable single-interval entanglement data. This is a property of the information content of the data, not a failure of the optimizer.
- **Multi-step collapses are harder.** For profiles that rise in two or three distinct stages, the intermediate plateaus are only partially resolved. Part of this is the blind spot above, and part is the shape of function the network is built to represent. Breaking the ambiguity likely requires richer boundary observables or stronger physical priors on the profile.
