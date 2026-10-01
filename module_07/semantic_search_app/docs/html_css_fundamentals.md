# HTML and CSS Fundamentals

HTML (HyperText Markup Language) describes the structure of a web page. Elements are written as tags such as <h1> for a top-level heading, <p> for a paragraph, <a href="..."> for a link, and <img src="..." alt="..."> for an image. Semantic elements like <header>, <nav>, <main>, <article>, and <footer> describe the purpose of each region, which helps screen readers and search engines understand the page.

Forms collect user input. A <form> element contains inputs such as <input type="text">, <input type="email">, <select>, and <textarea>, each with a name attribute that becomes the key when the form is submitted. A <label for="..."> tied to an input's id makes the form accessible and lets users click the label to focus the field. The action and method attributes decide where and how the data is sent.

CSS (Cascading Style Sheets) controls presentation. A rule has a selector and a block of declarations, for example p { color: #333; line-height: 1.5; }. Selectors can target element types, classes (.card), IDs (#main), attributes, and states such as :hover. When several rules match, specificity and source order decide which declaration wins, which is the "cascade" in the name.

The box model explains how size is calculated. Every element has content, padding, border, and margin. By default width applies only to the content box, so adding padding makes the element wider; setting box-sizing: border-box makes width include padding and border, which is easier to reason about and is common in modern resets.

Flexbox and Grid handle layout. display: flex arranges children along one axis with properties such as justify-content, align-items, and gap, which is ideal for navigation bars and rows of buttons. display: grid defines rows and columns in two dimensions with grid-template-columns, making it well suited to card galleries and full page layouts. Media queries such as @media (max-width: 600px) adapt the layout for small screens, which is the basis of responsive design.
