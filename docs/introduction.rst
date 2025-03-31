
============
Introduction
============

fratoo is an open-source multi-scale modelling framework based on `OSeMOSYS <http://www.osemosys.org/>`_. It facilitates the development and analysis of multi-scale energy system models and pathways.

**********
Background
**********

Energy system models are vital tools for analysing the energy system and its future evolution. Scenario pathways derived by energy system optimization models are widely used to support strategic thinking about energy futures, derive favourable energy system pathways, and support decision-making in energy planning processes.

A range of open-source modelling frameworks have been developed over recent years to support energy modellers and planners in the development of energy system optimization models, e.g., `OSeMOSYS <http://www.osemosys.org/>`_, `Temoa <https://temoacloud.com/>`_,  `Calliope <https://www.callio.pe/>`_, or `URBS <https://github.com/tum-ens/urbs>`_.


*******
Concept
*******

fratoo complements this existing pool of energy system optimization frameworks with a focus on facilitating the development of **multi-scale models** to derive scenario pathways that can support multi-scale or polycentric governance of sustainable energy transitions. As such, it is less of a new stand-alone framework but acts as a *frame* around OSeMOSYS.

fratoo aims to add to the existing landscape of frameworks, in particular OSeMOSYS, by pursuing following design goals:

* To allow for a straightforward and logical way to define and model energy systems across multiple spatial scales
* To facilitate versatile model runs of different spatial entities and scales, flexible (dis)aggregation of spatial entities and/or results
* To support prompt analysis of multi-scale results through suitable visualizations

.. note::  To avoid ambiguity, we here refer to *spatial entities* instead of *regions*. While *regions* often refer to separate, not overlapping areas (potentially on the \'regional\' scale), we here use the term *spatial entities*, which refers to spatial areas on any scale, e.g., a local area or an entire country, which can be part of or encompass other entities.

The diagram below gives an overview of general functionality of fratoo:

.. figure:: figures/framework.*
   :alt: verview of fratoo's functionality.

   Overview of fratoo's functionality.

fratoo serves as a frame around OSeMOSYS for processing input data, running scenarios, and processing results. It relies on two main inputs:

* a multi-scale input data set (as further explained :doc:`here <building>`)
* an OSeMOSYS Pyomo model file (for example to be found on the `OSeMOSYS github page <https://github.com/OSeMOSYS/OSeMOSYS>`_)

Energy systems can be modelled across an arbitrary number of spatial entities (e.g., local areas or regions), which in turn can be defined across an arbitrary number of scales (e.g., local or national scale). Each entity is part of a particular scale and, except the overarching entity (in the example below the UK), has usually a parent entity it is part of on an upper scale. An example structure is shown in the diagram below.

.. figure:: figures/multi-scale_structure.*
   :alt: Exemplary multi-scale structure.
   :width: 400

   Exemplary multi-scale structure with 3 different scales (national, regional, local) and 7 - and counting - spatial entities: United Kingdom (national scale), England, Wales, Scotland, and Northern Ireland (regional scale), Brighton, Camden, \.\.\. (local scale). The arrows are symbolizing the child/parent relation between entities (and not potential endogenous interactions between entities, which can be introduced between any entities).

This structure allows for different elements of the energy system to be modelled at different spatial scales based on the required detail or available data. For example, a building heat sector model looking at heat decarbonization in local areas across the country might include a power sector module represented on the national scale.

Based on input data and model file, fratoo allows the flexible generation and optimization of (sub)models, e.g., particular single spatial entities, aggregated entities, or sets of entities, and the post-processing and visualization of results (this is further explained in the :doc:`running` section).

.. [Explain specifically what's multi-scale about fratoo?]
.. multiscale definition, multi-scale pathways possible because of versatility of model runs, ...


*******
Licence
*******

fratoo is made available under a `MIT licence <https://opensource.org/licenses/MIT>`_:

    Copyright 2025 Leonhard Hofbauer

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

